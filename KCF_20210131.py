'''
In new platform (Python3.9) the original KCF_status_motion_vector runs very slowly.
Therefore, this file uses new version KCF (mainly using numpy  and scipy package) to perform FFT
and corss correlation.
Also add new kernel function options (Gaussian kernel and inner product kernel).
Created by Yi ZHOU on 20210131@Provence_Dalian.
'''

import cv2
import numpy as np
from scipy import signal        #   2d cross correlation
import matplotlib.pyplot as plt
import utilities_200611         as uti            # personal tools

# Keep reference template.
# Save Trajectories, estimated template, psr and regression matrix's maximum.


class KCFTracker(object):
    """
    define the tracker model, inclding init(), update() member functions.
    """
    # Init function is different for varied trackers, default kernel is 'Gaussian Kernel'
    # Kernel_sigma is a critical parameters to measure the similarity between reference and tested template.
    # For radar image processing, near range and far range get different image resolution, kernel_sigma should be
    # tuned to fit the resolution and get proper PSR.
    def __init__(self, img, rect, frame_no, kernel_opt='gk', kernel_sigma = 1.2):

        # set the kernel option('gk' for Gaussion kernel and 'ip' for inner_product)
        # for computing kernel matrix.
        self.kernel_opt   = kernel_opt
        self.kernel_sigma = kernel_sigma    #also called  'gaussian kernel bandwidth'

        ys = int(rect[1]) + np.arange(rect[3], dtype = int)
        xs = int(rect[0]) + np.arange(rect[2], dtype = int)

        self.imgh, self.imgw = img.shape[:2]
        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= img.shape[0]] = img.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1

        #self.rect = rect #rectangle contains the bounding box of the target
        #pos is the center postion of the tracking object (cy,cx)
        self.pos = np.array([rect[1] + rect[3]/2, rect[0] + rect[2]/2])

        self.posOffset = np.array([0,0], dtype = int)
        # parameters according to the paper --

        padding = 1.0  # extra area surrounding the target(扩大窗口的因子，默认扩大2倍)
        # spatial bandwidth (proportional to target)
        output_sigma_factor = 1 / float(16)
        self.lambda_value = 1e-2  # regularization

        # linear interpolation factor for reference updating.
        self.interpolation_factor = 0.075

        #target_ze equals to [rect3, rect2]
        self.target_sz = np.array([int(rect[3]), int(rect[2])])
        # window size(Extended window size), taking padding into account
        self.window_sz = np.int0(self.target_sz * (1 + padding)) #Search region is twice bigger than target size
        self.tly, self.tlx = self.pos - np.int0(self.window_sz / 2) # topleft x and y for current template.

        # store the initial_target_template, frame_no, rect for SVM training samples.
        # This is memory consuming for big numbers of targets.
        self.target_template = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        # desired output (gaussian shaped), bandwidth proportional to target size
        output_sigma = np.sqrt(np.prod(self.target_sz)) * output_sigma_factor
        grid_y = np.arange(self.window_sz[0]) - np.floor(self.window_sz[0] / 2)
        grid_x = np.arange(self.window_sz[1]) - np.floor(self.window_sz[1] / 2)
        # [rs, cs] = ndgrid(grid_x, grid_y)
        rs, cs = np.meshgrid(grid_x, grid_y)
        dist2  = rs ** 2 + cs ** 2

        dist2[dist2>5000] = 5000
        try:
            y = np.exp(-0.5* dist2 / (np.spacing(1)+output_sigma ** 2 ))
        except Exception as e:
            print(e)

        self.yf= np.fft.fft2(y)
        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.window_sz[0]), np.hanning(self.window_sz[1]))
        # get subwindow at current estimated target position, to train classifer
        x = self.get_subwindow(img, self.pos, self.window_sz, self.cos_window)
        # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
        K = self.get_kernel(x)
        #storing computed alphaf and z for next frame iteration
        self.alphaf = np.divide(self.yf, (np.fft.fft2(K) + self.lambda_value))  # Eq. 7
        self.z = x

        self.psr = self.response_psr(y, cx=np.floor(self.window_sz[1] / 2),
                                        cy=np.floor(self.window_sz[0] / 2), winsize=12)
        self.response = y
        self.target_rect = rect
        self.init_frame_no   = frame_no
        self.tracked_Frames  = 0
        self.ypeak_list = []        # a list to store the region matrix's peak value
        self.psr_list   = []
        self.trajectories = {frame_no:rect}      # traj dict{fid:rect}
        #return initialization status
        #return True

    def get_kernel(self, x, y=None):
        '''
        compute the kernel matrix based on the correlation results of the reference template
        and the circular shifted results.
        :param kernel_opt:
        :return:
        '''
        """
                Gaussian Kernel with dense sampling.
                Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
                between input images X and Y, which must both be MxN. They must also
                be periodic (ie., pre-processed with a cosine window). The result is
                an MxN map of responses.

                If X and Y are the same, omit the third parameter to re-use some
                values, which is faster.
        """
        xf = np.fft.fft2(x)  # x in Fourier domain
        xx = np.sum(x ** 2)
        if y is not None:
            # general case, x and y are different
            yf = np.fft.fft2(y)
            yy = np.sum(y ** 2)
        else:
            # auto-correlation of x, avoid repeating a few operations
            yf = xf
            yy = xx
            y = x
        # cross-correlation term in Fourier domain
        xyf = np.multiply(xf, np.conj(yf)) #element-wise product
        # to spatial domain
        xyf_ifft = np.fft.ifft2(xyf)
        #xy_complex = circshift(xyf_ifft, floor(x.shape/2))
        row_shift, col_shift = np.int0(np.array(x.shape)/2)
        xy_complex = np.roll(xyf_ifft,  row_shift,  axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        xy = np.real(xy_complex)

        if self.kernel_opt == 'gk': # Gaussian kernel
            sigma  = self.kernel_sigma
            #sigma = 1.2    # for titan
            #sigma = 0.04   # for Gaussian ETT
            #sigma = 1      # fro Trifalo

            scaling = -1 / (sigma ** 2)
            xx_yy_2xy = xx + yy - 2 * xy
            Kmatrix = np.exp(scaling * np.maximum(0, xx_yy_2xy / (x.size**2)))
        else: # using innper product kernel
            #Kmatrix = signal.correlate2d(x, y, boundary='circular', mode='same')
            Kmatrix = xy
        return Kmatrix

    def get_subwindow(self, im, pos, sz, cos_window):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).

        normalize helps to resist the changes of illumination.
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.int0(pos[0] + np.arange(sz[0], dtype=int) - int(sz[0]/2))
        xs = np.int0(pos[1] + np.arange(sz[1], dtype=int) - int(sz[1]/2))

        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        #zs = range(im.shape[2])

        # extract image
        #out = im[pylab.ix_(ys, xs, zs)]
        out = im[np.ix_(ys, xs)]

        # if debug:
        #     print("Out max/min value==", out.max(), "/", out.min())
        #     plt.figure()
        #     plt.imshow(out, cmap=plt.cm.gray)
        #     plt.title("cropped subwindow")

        #pre-process window --
        # normalize to range -0.5 .. 0.5
        # pixels are already in range 0 to 1
        out = out.astype(np.float64) - 0.5
        #out = uti.frame_normalize(out)
        # apply cosine window
        out = np.multiply(cos_window, out)

        return out

    def  update(self, new_img, frame_no):
        '''
        :param new_img: new frame should be normalized, for tracker_status estimating the rect_snr
        :return:
        '''

        self.tracked_Frames +=1
        # get subwindow at current estimated target position, to train classifier
        x = self.get_subwindow(new_img, self.pos, self.window_sz, self.cos_window)
        # calculate response of the classifier at all locations
        K = self.get_kernel(x, self.z)
        Kf = np.fft.fft2(K)
        alphaf_kf = np.multiply(self.alphaf, Kf)
        response  = np.real(np.fft.ifft2(alphaf_kf))     # Eq. 9
        #response  = uti.frame_normalize(response)        # Normalize from 0 to 1
        #response  = response/np.sum(response)
        self.response=response
        #self.kmax = np.max(K)
        self.responsePeak = np.max(response)
        # target location is at the maximum response
        row, col = np.unravel_index(response.argmax(), response.shape)
        #row, col = np.unravel_index(K.argmax(), K.shape)
        #roi rect's topleft point add [row, col]
        #self.tly, self.tlx = np.max(0, self.pos - np.int0(self.window_sz / 2))
        self.tly, self.tlx = self.pos - np.int0(self.window_sz / 2)

        #here the pos is not given to self.pos at once, we need to check the psr first.
        #if it above the threashhold(default is 5), self.pos = pos.
        pos = np.array([self.tly, self.tlx]) + np.array([row, col])
        rx, ry, rw, rh =[pos[1]- self.target_sz[1]/2, pos[0] - self.target_sz[0]/2, self.target_sz[1], self.target_sz[0]]
        # limit the rect inside the image
        rx  = min(max(0, rx), self.imgw-1) # 0 <= rx <= imgw-1
        ry  = min(max(0, ry), self.imgh-1)
        bx  = rx + rw
        by  = ry + rh
        bx  = min(max(0, bx), self.imgw-1) # 0 <= bx <= imgw-1
        by  = min(max(0, by), self.imgh-1)
        rw  = int(bx - rx)
        rh  = int(by - ry)
        #assert(rw>0 and rh>0)

        #rect = np.array([pos[1]- self.target_sz[1]/2, pos[0] - self.target_sz[0]/2, self.target_sz[1], self.target_sz[0]])
        rect = np.array([rx,ry, rw, rh])
        rect = rect.astype(np.int)

        self.target_rect = rect
        self.psr = self.response_psr(response, col, row, winsize=12)
        self.pos = pos


        #only update when tracker_status's psr is high
        if (self.psr > 10):
        #if (self.responsePeak>0.3):
            #computing new_alphaf and observed x as z
            x = self.get_subwindow(new_img, pos, self.window_sz, self.cos_window)
            # Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
            k = self.get_kernel(x)
            new_alphaf = np.divide(self.yf, (np.fft.fft2(k) + self.lambda_value))  # Eq. 7
            new_z = x
            # subsequent frames, interpolate model
            f = self.interpolation_factor
            self.alphaf = (1 - f) * self.alphaf + f * new_alphaf
            self.z = (1 - f) * self.z + f * new_z

        if rw<=0 or rh<=0: #try eliminate the target on the boundary.
            print('Negative w/h - ',' psr ', self.psr, ' peak ', self.responsePeak )
            self.psr = 0
            self.responsePeak = 0
        self.target_rect = rect
        self.trajectories[frame_no] = rect
        self.ypeak_list.append(self.responsePeak)
        self.psr_list.append(self.psr)
        #return ok, rect, self.psr, response # for mkcf
        return rect, self.psr, self.responsePeak#, response

    def response_psr(self, response, cx, cy, winsize):
        '''
        computing the average and maximum value in a monitor window of response map
        :param response:
        :param cx:
        :param cy:
        :return:
        '''
        # res_monitor_windows_size
        tlx = int(max(0, cx - winsize / 2))
        tly = int(max(0, cy - winsize / 2))
        brx = int(min(cx + winsize / 2, response.shape[1]))
        bry = int(min(cy + winsize / 2, response.shape[0]))

        reswin = response[tly:bry, tlx:brx]
        res_win_max = np.max(reswin)

        sidelob = response.copy()
        exclude_nums = (brx-tlx)*(bry-tly)
        #excluding the peak_neighbour
        sidelob[tly:bry, tlx:brx] = 0
        #print(sidelob.shape)
        sidelob_mean = np.sum(sidelob)/(sidelob.shape[0]*sidelob.shape[1] - exclude_nums)

        sidelob_var  = (sidelob - sidelob_mean)**2
        #exclude the peak_neighbour
        sidelob_var[tly:bry, tlx:brx] = 0
        sidelob_var = np.sum(sidelob_var)/(sidelob.shape[0]*sidelob.shape[1] - exclude_nums)

        #peak to sidelobe ratio
        psr = (res_win_max-sidelob_mean)/np.sqrt(sidelob_var+np.spacing(1))
        return  psr