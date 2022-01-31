import sys
sys.path.append("../segmentation/")  #for import the utility in up-directory/segmention/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Local Binary Pattern function
# from skimage.feature import local_binary_pattern
# from skimage import transform
#import utility as uti
import cv2

def segmentation(frame, lbp_contrast_select = False, kval=1, least_wh = (3,3), min_area=32, max_area=200, nref=25, mguide=18, roi_mask=np.array([])):
    # kval decide the false alarm rate of cfar.
    # least_wh is the least window size of the segmentation
    # using contrast intensity or not. Should not be used, this is not adaptive threshold
    cfar_cs    = CFAR(kval=kval, nref=nref, mguide=mguide)
    bin_image  = cfar_cs.cfar_seg(frame)
    if roi_mask.size>0:
        bin_image = bin_image*roi_mask
    (contours, _) = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #(contours, _) = cv2.findContours(bin_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    blob_bb_list  = []  #blob bounding box list

    if lbp_contrast_select: # Eliminate blobs with low contrast intensities
        inv_var = local_binary_pattern(frame, P=24, R=3, method='var')
        inv_var[np.isnan(inv_var)] = 0
        int_img = transform.integral_image(inv_var)

    for id, contour in enumerate(contours):
        x, y, w, h= cv2.boundingRect(contour)  # only using the bounding box information
        # rotated_rect = cv2.fitEllipse(cnt)
        # rotatedBox   = cv2.minAreaRect(approx_contour)
        bb_rect = [x,y,w,h]
        # omit the 20meters under the radar
        if (y <= 10):
            continue
        # omit too small or too big rectangles.
        if (w*h <= min_area) or (w*h>=max_area)or (w<=least_wh[0]) or (h<=least_wh[1]):
            continue
        if lbp_contrast_select:
            iiv = transform.integrate(int_img, (y, x), (h + y - 1, x + w - 1))
            # omit the sparse density of variance image.
            if (iiv[0] / (w * h)) < 500:
                continue
        blob_bb_list.append(bb_rect)
    return blob_bb_list, bin_image

class CFAR(object):

    #def __init__(self, kval = 1.0, nref=8, mguide=4):
    def __init__(self, kval=1.0, nref=32, mguide=16):
        self.kval   = kval  # decide the kvalue
        self.nref   = nref   # number of reference cell
        self.mguide = mguide    # number of guide cell

    def cfar_ave(self, echos, nref=8, mguide=4):
        '''
        Constant false alarm rate on the echos, only computing the average of the reference cells
        Please note the azimuth units equals to the width.
        Assumption, raylay noise,
        threshold = 4.29, 4.80, 5.26 related to the false alarm 10-4, 10-5, 10-6.
        :param echos:  input radar echos
        :param nref:   reference ceil numbers
        :param mguide: guide ceil numbers
        :return: average of the reference cells.
            A0 A1 ... A(azimuth_units-1)
        R0  x  x      x
        R1  x  x      x
        .
        .
        .
        R(rang_units-1)
                                   pos, [guide],      [reference]
        Sa: sum after n reference.  i, [i+1,...,i+m], [i+m+1,...,i+m+n]
                                        [reference],       [guide]        pos
        Sb: sum before  n reference.  [i-m-n, ..., i-m-1], [i-m, ..., i-1], i

        when  0  <=i<= range_units-1-m-n : Sa exists.
        when  m+n<=i<= range_units-1     : Sb exists.
        when  m+n<=i<= range_units-1-m-n : Both Sb and Sa exists.

        when 0 <= i < m+n :  only Sb exists. Ave = Sa/n
        when m+n<=i<=range_units-1-m-n   :   Ave = (Sb + Sa)/2n
        when range_units-1-m-n <= i <= rang_units -1 : only Sa exits Ave = Sb/n
        '''
        range_units, azimuth_units = echos.shape
        #range_units = echos.shape[0]

        ave_ref = np.zeros_like(echos, dtype='float')
        m = mguide
        n = nref

        # Initialize sum after
        Sa = np.sum(echos[m + 1: m + n + 1, :], 0)  # sum on the column direction, from row (m+1 to m+n)
        # Initialize sum before
        Sb = np.sum(echos[0:n, :], 0)  # sum on the column direction, from row (0 to n-1)

        # left beginning n+m cells
        for i in range(range_units):
            # updating Sa and Sb row by row.
            if 1 <= i <= range_units - m - n - 1:  # omit i=0, using initialized Sb
                Sa = Sa + echos[i + m + n, :] - echos[i + m, :]
            if m + n <= i <= range_units - 1:
                Sb = Sb + echos[i - m, :] - echos[i - m - n, :]  # omint i==m+n, using initialized St

            if 0 <= i < m + n:
                ave_ref[i, :] = Sa / n
            if m + n <= i <= range_units - m - n - 1:
                ave_ref[i, :] = (Sb + Sa) / (2 * n)
            if range_units - m - n - 1 < i < range_units:
                ave_ref[i, :] = Sb / n

        return ave_ref


    def cfar_thresh(self, echos, ave_ref, kval):
        '''
        get the echos which has higher value than the threahold (ave_ref*kval)
        :param echos:
        :param ave_ref:
        :param kval:
        :return:
        '''
        cfar_mask = (echos >= ave_ref * kval) * 1
        cfar_echos = echos * cfar_mask
        return cfar_mask, cfar_echos

    def set_parameters(self, kval=1.0, nref=8, mguide=4):
        self.kval   = kval
        self.nref   = nref
        self.mguide = mguide

    def cfar_seg(self, echos):
        '''
        echos get range in the vertical direction, get azimuth on the horizontal direction.
        :param echos:
        :return: segmented binary image, 1 means object, 0 means background.
        '''
        ave_vertical   = self.cfar_ave(echos, self.nref, self.mguide)
        #decreasing the value in the horizontal direction.
        #ave_horizontal = self.cfar_ave(echos.T, int(self.nref-8), int(self.mguide-8))
        ave_horizontal = self.cfar_ave(echos.T, int(self.nref - self.nref/2), int(self.mguide - self.mguide/2))
        mask_vertical,     cfar_echos_vertical   = self.cfar_thresh(echos, ave_vertical, self.kval)
        mask_horizontal,   cfar_echos_horizontal = self.cfar_thresh(echos.T, ave_horizontal, self.kval)
        # self.ax_cfar_range.imshow(self.mask_range)
        # self.ax_cfar_azi.imshow(self.mask_azi.T)
        mask = mask_vertical + mask_horizontal.T
        #self.ax_cfar_mask.imshow(self.mask)
        # area with both mask_range and mask_azi has value 2.
        bin_image = (mask == 2) * 1
        bin_image = bin_image.astype('uint8')
        #self.draw_plt_polyline(bin_image)
        return bin_image

    def draw_plt_polyline(self, bin_image):
        blob_list = uti.contour_extract(bin_image)
        fig, ax = plt.subplots()
        plt.imshow(self.echos, cmap='jet')
        for i, blob_reg in enumerate(blob_list):
            poly_xys = np.array(blob_reg['Polygon'])
            xs = poly_xys[:, 0]
            ys = poly_xys[:, 1]
            line = Line2D(xs, ys, marker='o', markerfacecolor='r', color='black')
            ax.add_line(line)
            cx,cy = blob_reg['Center']
            ax.text(cx, cy, str(i))
        plt.show()
        #self.polyregion_histogram(bin_image, self.echos)

    def draw_cv_polyline(self, bin_image, echos):
        canvas = cv2.cvtColor(echos, cv2.COLOR_GRAY2BGR)
        (contours, _) = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours,contourIdx=-1, color=[0,255,0])
        # cv2.imshow('cfar_seg', canvas)
        # cv2.waitKey()
        return canvas

from PIL import Image
import glob

if __name__=='__main__':
    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    #test_frame = np.array(Image.open('%s/%02d.png' % (file_prefix, frame_no)))
    file_names = glob.glob(file_prefix+'*.png')
    file_names.sort()
    # cv2.namedWindow('inesa', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('inesa', 1024, 768)
    # cv2.moveWindow('inesa', 200, 100)
    cfar = CFAR(kval=1)
    bGo=True
    while(bGo):
        for img_file in file_names:
            frame_no = img_file.split('/')[-1].split('.')[0]
            frame = np.array(Image.open(img_file))
            bin_img = cfar.cfar_seg(frame)
            canvas  = cfar.draw_cv_polyline(bin_img, frame)
            #cv2.putText(frame, frame_no, (500, 100), 2,2, (0,255,0))
            cv2.setWindowTitle('inesa', str(frame_no))
            cv2.imshow('inesa', canvas)
            if(cv2.waitKey()& 0xFF == ord('q')):
                bGo=False
                break