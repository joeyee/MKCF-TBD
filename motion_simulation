'''
Simulate motion model for the point-target and extended-target in clutter.
Motion model: constant velocity (CV), constant acceleration (CA), constant turn (CT).

Fluctuating Extended targets model: Swerling targets of type 0,1,3.
Created by Yi Zhou on 20201030 @Provence-Dalian.
Add  swerling target model on 20210302 @Provence-Dalian.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,rayleigh
import numpy.linalg as la
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import kalman_filter_20201029 as kf_model
import jpda_IJOE2016_20201029 as jpda_model

from scipy import ndimage
from scipy.stats import norm,rayleigh
from scipy.ndimage.interpolation import rotate
from scipy.stats                 import norm,rayleigh,chi,chi2

import utilities_200611         as uti            # personal tools

def constant_velocity(x0, y0, velo, npoints):
    '''
    :param x0: start point's x
    :param y0: start point's y
    :param velo: constant velocity(vx,vy)
    :param npoints: number of points
    :return:
    '''
    ts = np.linspace(0, npoints-1, npoints)
    vx,vy = velo
    xs = x0 + vx*ts
    ys = y0 + vy*ts
    return xs, ys

def constant_accelerate(x0, y0, velo, acc, npoints):
    '''
    :param x0: start point's x
    :param y0: start point's y
    :param velo: initial velocity(vx,vy)
    :param acc: constant accelerate
    :param npoints: number of points
    :return: trajectory of xs, ys
    '''
    ts = np.linspace(0, npoints-1, npoints)
    vx,vy = velo
    ax,ay = acc
    xs = x0 + vx*ts + (ts**2)*ax/2
    ys = y0 + vy*ts + (ts**2)*ay/2
    return xs, ys

def constant_turn(x0, y0, radius, omega, npoints):
    '''
    :param x0: start point's x
    :param y0: start point's y
    :param radius: radius of turning
    :param omega: constant turning rate
    :return:trajectory
    '''

    ts = np.linspace(0, npoints-1, npoints)
    xs = x0 + np.sin(omega*ts)*radius
    ys = y0 + np.cos(omega*ts)*radius
    return xs, ys

def get_orientation(xs,ys):
    '''
    Get velocity based on the locations,
    Using the velocity to compute the orientation.
    :param xs:
    :param ys:
    :return:
    '''
    dys = np.diff(ys)
    dxs = np.diff(xs)
    #compute the orientation of the extended target by velocity.
    thetas_less = np.arctan2(dys, dxs)  # len(dxs) - 1
    thetas      = np.pad(thetas_less, (0, 1), 'edge')  # add one elements to the end
    return thetas


def s_manuver():
    '''
    a S-type manuvering trajectory: including a cv, ca, cv and ct.
    :return: trajectories
    '''
    x0  = 10
    y0  = 30
    #velo= (2*2, -1*2)
    velo = (5 * 2, -1 * 2)
    npoints = 10
    x1s, y1s = constant_velocity(x0, y0, velo, npoints)

    x0= x1s[-1]
    y0= y1s[-1]
    #velo = (1*2,1*2)
    velo = (2 * 2, 4 * 2)
    acc = (-0.25,1)
    npoints = 8
    x2s, y2s = constant_accelerate(x0, y0, velo, acc, npoints)

    x0 = x2s[-1]
    y0 = y2s[-1]
    #velo = (1*2, 1*2)
    velo = (5 * 2, 2 * 2)
    npoints = 10
    x3s, y3s = constant_velocity(x0, y0, velo, npoints)

    radius = 30
    omega =  0.3
    npoints =12
    x0 =  x3s[-1]+4 - radius*np.sin(omega)
    y0 =  y3s[-1] - radius*np.cos(omega)
    x4s,y4s = constant_turn(x0, y0, radius, omega, npoints)

    xs = x1s.tolist() + x2s.tolist() + x3s.tolist() +  x4s.tolist()
    ys = y1s.tolist() + y2s.tolist() + y3s.tolist() +  y4s.tolist()

    fig, ax = plt.subplots()
    w1t = 15
    h1t = 9
    npoints = len(xs)
    ws = np.random.normal(w1t, 0.5, npoints)
    hs = np.random.normal(h1t, 0.5, npoints)

    dys = np.diff(ys)
    dxs = np.diff(xs)
    #compute the orientation of the extended target by velocity.
    thetas_less = np.arctan2(dys, dxs)  # len(dxs) - 1
    thetas      = np.pad(thetas_less, (0, 1), 'edge')  # add one elements to the end

    # # visualize the trajectory of the extended target
    plot_ellipse(ax, xs, ys, ws, hs, facecolor='green')
    plt.show()
    #
    # tx = [str(i) for i in range(1,len(xs)+1)]
    # show_text(xs, ys, tx)          #show text
    # plot_trajectory(xs,ys,'green') #draw trajectory
    return xs,ys,ws,hs,thetas


'''
# This is an example from the ndimage, to compute the Gaussian kernel.
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    p = numpy.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(p(x), dtype=numpy.double)
    phi_x /= phi_x.sum()
    if order > 0:
        q = numpy.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)
    return phi_x
'''

def gaussian_kernel2d(sigma_x, sigma_y, theta, bnorm=True):
    '''
    Return a 2d Gaussian kernel template (2d matrix).
    :param sigma_x:
    :param sigma_y:
    :param theta: rotation theta of 2d Gaussian
    :return: Gaussian Kernel Template.
    '''
    kernel_wr = np.int(sigma_x * 2.5 + 0.5)
    kernel_hr = np.int(sigma_y * 2.5 + 0.5)

    #if kernel_hr < 5 or kernel_wr < 5:

    #    raise ValueError('kenrel width or/and height are too small')

    kx = np.arange(-kernel_wr, kernel_wr + 1)
    ky = np.arange(-kernel_hr, kernel_hr + 1)
    KX, KY = np.meshgrid(kx, ky)
    theta = -1*theta

    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta)  / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    # f(x,y)=Aexp(−(a(x−xo)2+2b(x−xo)(y−yo)+c(y−yo)2)) , here xo=0, yo=0
    # f(x,y)=Aexp(−(ax^2+2bxy+cy^2))
    # a   = cos2θ2σ2X + sin2θ2σ2Y
    # b   =−sin2θ4σ2X + sin2θ4σ2Y
    # c   = sin2θ2σ2X + cos2θ2σ2Y

    kgauss = np.exp(-(a * KX ** 2 + 2 * b * KX * KY + c * KY ** 2))
    if bnorm:#normalization in default mode.
        kgauss = kgauss / np.sum(kgauss)
    return kgauss

local_snrs  = []
global_snrs = []
#constant_template = gaussian_kernel2d(8,4,0)
dec_str = [12  ,  11  , 10  , 9   , 8   , 7   , 6   , 5   , 4   , 3   , 2   , 1   , 0   , -1  , -2  ]
def add_gaussian_template_on_clutter(cx, cy, w, h, theta, erc, snr, clutter_background, swerling_type=0):
    # Erc: average clutter energy.
    # Erc = np.sum(clutter_background ** 2) / clutter_background.size
    sigma_x = (w / 2.5 - 0.5) / 2 #sigma_x is related to the width of the template
    sigma_y = (h / 2.5 - 0.5) / 2

    kgauss = gaussian_kernel2d(sigma_x, sigma_y, theta) # Get diffusive coefficients for a 2d gaussian
    # kh_big,kw_big      = kgauss_big.shape[:2]
    # kh,kw      = [int(kh_big/2), int(kw_big/2)]
    # kly,klx    = [int(kh/2),     int(kw/2)]
    # kgauss     = kgauss_big[kly:kly+kh, klx:klx+kw]
    Egk_numer = np.sum(kgauss.ravel() ** 2) / kgauss.size # 2d gaussian's average power.

    h_t, w_t = kgauss.shape
    ly = int(cy - (h_t - 1) / 2)
    ry = int(cy + (h_t - 1) / 2)
    lx = int(cx - (w_t - 1) / 2)
    rx = int(cx + (w_t - 1) / 2)

    img_h, img_w = clutter_background.shape
    if ly < 0 or lx < 0 or ry > img_h or rx > img_w:
        raise ValueError('template location is beyond the image boundaries!')
    bk_roi = clutter_background[ly:ly + h_t, lx:lx + w_t]
    # compute the amplitude coefficients according to the SNR Eq.
    kcoef_global = np.sqrt(np.power(10, (snr / 10)) * erc / Egk_numer)
    # average power of clutter is computed by numerical results in local roi-window.
    erc_local    = np.sum(bk_roi ** 2) / bk_roi.size
    kcoef_local  = np.sqrt(np.power(10, (snr / 10)) * erc_local / Egk_numer)

    kcoef = kcoef_global
    if swerling_type == 0:  # swerling type 0 target
        kcoef_t  = kcoef
        template = kgauss * kcoef_t
    if swerling_type == 1:
        sigma    = kcoef  # /np.sqrt(2)
        # central amplitude obeys the rayleigh distribution, which 2*sigma^2 = sigma_t = kcoef**2 (swerling_0's Amplitude)
        kcoef_t  = rayleigh.rvs(loc=0, scale=sigma, size=1)
        template = kgauss * kcoef_t
    if swerling_type == 3:  # central amplitude obeys the chi distribution, which degrees of freedom k=4.
        kcoef_t =  chi2.rvs(df=kcoef, size=1)  # or kcoef_t  = chi2.rvs(df=kcoef, size=1), then template=kgauss*kcoef
        template = kgauss * (kcoef_t)  # for chi2, Mean=df.

    # Get decrease_coeffient to make sure the inner gaussian template satisfy the snr requirement.
    tcx, tcy = w_t/2, h_t/2
    snr_lis= list(range(12, -3, -1))  # [12, 11, ..., -1, -2]
    # shrink rate, take from cfar results.
    snr_lis= [12  ,  11  , 10  , 9   , 8   , 7   , 6   , 5   , 4   , 3   , 2   , 1   , 0   , -1  , -2  ]
    wr_lis = [1.62,  1.67, 1.65, 1.76, 1.80, 2.00, 2.20, 2.30, 3.20, 3.50, 3.70, 3.90, 4.00, 4.2, 4.5]
    hr_lis = [0.88,  0.89, 0.90, 0.92, 1.00, 1.10 ,1.20, 1.20, 1.55, 1.55, 1.65, 1.70, 1.75, 2.0, 2.5]
    decs   = [0.77,  0.76, 0.75, 0.74, 0.73, 0.66, 0.62, 0.61, 0.50, 0.48, 0.42, 0.38, 0.35, 0.28,0.25]
    #decrease the size of Gaussian template, similar to the cfar_seg results.
    # [cfar shrink the real target, when outside is lower than center]
    wr = wr_lis[snr_lis.index(snr)]
    hr = hr_lis[snr_lis.index(snr)]
    iw,  ih  = w_t/wr,  min(h_t/hr, h_t)
    ix, iy, iw, ih = np.int0([tcx-iw/2, tcy-ih/2, iw, ih])
    inner_gauss    = template[iy:iy+ih, ix:ix+iw]


    dec_coef       = np.sqrt(np.power(10, (snr / 10)) * erc_local / np.mean(inner_gauss**2))
    dec_str[snr_lis.index(snr)] = '%.2f'%dec_coef

    dec_coef       = decs[snr_lis.index(snr)]
    template =  template*dec_coef #np.sqrt(1.618) #/2.8 # Make sure that in shrinked (cfar-segmented) target region still holds low snr.
    loc_snr  = 10 * np.log10(np.sum(template ** 2) / np.sum(bk_roi ** 2))
    glob_snr = 10 * np.log10(np.sum(template ** 2) / (erc * template.size))
    # print('Swerling Type %d, kcoef_t %.2f (w %d, h %d), extened_egk %.2E' % (swerling_type, kcoef_t, w, h, Egk_numer))
    # print('average (target - local clutter) power is (%.2f - %.2f)' % (np.sum(template ** 2) / template.size, erc_local))
    # print('Asked snr is %d, simulated local snr is %.2f, simulated global snr is %.2f' % (snr, loc_snr, glob_snr))
    #local_snrs.append(loc_snr)
    #global_snrs.append(glob_snr)
    mask = ([template > bk_roi]) * template
    clutter_background[ly:ly + h_t, lx:lx + w_t] = mask + bk_roi
    #clutter_background[ly:ly + h_t, lx:lx + w_t] = template + bk_roi
    return clutter_background

def add_gaussian_template_on_clutter_v2(cx, cy, w, h, theta, erc, snr, clutter_background, swerling_type=0):
    '''
    Rewrite the swerling type's pdf. kgauss is normalized.
    :return:
    '''
    # Erc: average clutter energy.
    # Erc = np.sum(clutter_background ** 2) / clutter_background.size
    sigma_x = (w/2  - 0.5) / 2  # sigma_x is related to the width of the template
    sigma_y = (h/2  - 0.5) / 2

    kgauss = gaussian_kernel2d(sigma_x, sigma_y, theta, bnorm=False)  # Get diffusive coefficients for a 2d gaussian
    Egk_numer = np.sum(kgauss.ravel() ** 2) / kgauss.size  # 2d gaussian's average power.

    h_t, w_t = kgauss.shape
    ly = int(cy - (h_t - 1) / 2)
    ry = int(cy + (h_t - 1) / 2)
    lx = int(cx - (w_t - 1) / 2)
    rx = int(cx + (w_t - 1) / 2)

    img_h, img_w = clutter_background.shape
    if ly < 0 or lx < 0 or ry > img_h or rx > img_w:
        raise ValueError('template location is beyond the image boundaries!')
    bk_roi = clutter_background[ly:ly + h_t, lx:lx + w_t]
    # compute the amplitude coefficients according to the SNR Eq.
    kcoef_global = np.sqrt(np.power(10, (snr / 10)) * erc / Egk_numer)

    kcoef_peak = np.sqrt(np.power(10, (snr / 10)) * erc) # point's snr reversion
    # average power of clutter is computed by numerical results in local roi-window.
    erc_local = np.sum(bk_roi ** 2) / bk_roi.size
    kcoef_local = np.sqrt(np.power(10, (snr / 10)) * erc_local / Egk_numer)

    kcoef = kcoef_peak
    if swerling_type == 0:  # swerling type 0 target
        kcoef_t  = kcoef
        template = kgauss * kcoef_t
    if swerling_type == 1:
        ray_scale = kcoef/np.sqrt(2)#choosing mode  # /np.sqrt(2)
        # central amplitude obeys the rayleigh distribution, which 2*sigma^2 = sigma_t = kcoef**2 (swerling_0's Amplitude)
        kcoefs = rayleigh.rvs(loc=0, scale=ray_scale, size=1000)
        kcoef_t = np.mean(kcoefs)
        template = kgauss * kcoef_t
    if swerling_type == 3:  # central amplitude obeys the chi distribution, which degrees of freedom k=4.
        df = 4
        chi2_scale= kcoef/np.sqrt(df*2+df**2)#np.sqrt(df-2)#
        kcoefs    = chi2.rvs(df=df, scale=chi2_scale,  size=1000)# or kcoef_t  = chi2.rvs(df=kcoef, size=1), then template=kgauss*kcoef
        kcoef_t   = np.mean(kcoefs)
        template  = kgauss * (kcoef_t) #

    # Get decrease_coeffient to make sure the inner gaussian template satisfy the snr requirement.
    tcx, tcy = w_t / 2, h_t / 2
    snr_lis = list(range(12, -3, -1))  # [12, 11, ..., -1, -2]
    # shrink rate, take from cfar results.
    snr_lis = [12,     11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,  -1,  -2]
    wr_lis  = [1.62, 1.67, 1.65, 1.76, 1.80, 2.00, 2.20, 2.30, 3.20, 3.50, 3.70, 3.90, 4.00, 4.2,  4.5]
    hr_lis  = [0.88, 0.89, 0.90, 0.92, 1.00, 1.10, 1.20, 1.20, 1.55, 1.55, 1.65, 1.70, 1.75, 2.0,  2.5]
    incs_sw1= np.linspace(1.00, 2.55, 15)#[0.95, 1.00, 0.90, 0.85, 0.80, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 2.00, 2.00, 2.20, 2.50]
    #incs_sw1 = np.log2(1+incs_sw1)
    decs    = np.linspace(0.78, 0.34, 15)
    #decs_sw1= np.linspace(1.00, 0.45, 15)
    decs_sw3= np.linspace(1.20, 0.30, 15)
    # decrease the size of Gaussian template, similar to the cfar_seg results.
    # [cfar shrink the real target, when outside is lower than center]
    wr = wr_lis[snr_lis.index(snr)]
    hr = hr_lis[snr_lis.index(snr)]
    iw, ih = w_t / wr, min(h_t / hr, h_t)
    ix, iy, iw, ih = np.int0([tcx - iw / 2, tcy - ih / 2, iw, ih])
    inner_gauss = template[iy:iy + ih, ix:ix + iw]

    #dec_coef = np.sqrt(np.power(10, (snr / 10)) * erc_local / np.mean(inner_gauss ** 2))
    #dec_str[snr_lis.index(snr)] = '%.2f' % dec_coef

    if swerling_type == 0: # decreasing for non-fluctuating target type
        dec_coef = decs[snr_lis.index(snr)]
        template = template * 1#dec_coef  # np.sqrt(1.618) #/2.8 # Make sure that in shrinked (cfar-segmented) target region still holds low snr.
    if swerling_type == 1:
        inc_coef = incs_sw1[snr_lis.index(snr)]
        template = template * 1   #inc_coef
    if swerling_type == 3:
        dec_coef = decs_sw3[snr_lis.index(snr)]
        template = template * 1#dec_coef
    loc_snr  = 10 * np.log10(np.sum(template ** 2) / np.sum(bk_roi ** 2))
    glob_snr = 10 * np.log10(np.sum(template ** 2) / (erc * template.size))
    peak_snr = 10 * np.log10(np.max(template)**2   /  erc) #point's snr

    # print('Swerling Type %d, kcoef_t %.2f (w %d, h %d), extened_egk %.2E' % (swerling_type, kcoef_t, w, h, Egk_numer))
    # print('average (target - local clutter) power is (%.2f - %.2f)' % (np.sum(template ** 2) / template.size, erc_local))
    # print('Asked snr is %d, simulated local snr is %.2f, simulated global snr is %.2f' % (snr, loc_snr, glob_snr))
    local_snrs.append(loc_snr)
    global_snrs.append(peak_snr)
    mask = ([template > bk_roi]) * template
    clutter_background[ly:ly + h_t, lx:lx + w_t] = mask + bk_roi
    #clutter_background[ly:ly + h_t, lx:lx + w_t] = template + bk_roi

    #Real_SNR is normally higher than peak_snr
    real_snr = 10 * np.log10(max(np.max(template + bk_roi)-np.sqrt(2), np.spacing(1)) / 2)

    return clutter_background

def add_uniform_template_on_clutter(cx, cy, w, h, theta, erc, snr, clutter_background, swerling_type=0):
    # Erc: average clutter energy.
    # Erc = np.sum(clutter_background ** 2) / clutter_background.size
    # Clutter_background is a clutter background template.

    kuniform  = np.ones((int(h),int(w)))/(h*w)
    unk_numer = np.sum(kuniform.ravel() ** 2) / kuniform.size    # 2d gaussian's average power.
    h_t, w_t = kuniform.shape
    ly = int(cy - (h_t - 1) / 2)
    ry = int(cy + (h_t - 1) / 2)
    lx = int(cx - (w_t - 1) / 2)
    rx = int(cx + (w_t - 1) / 2)

    img_h, img_w = clutter_background.shape
    if ly < 0 or lx < 0 or ry > img_h or rx > img_w:
        raise ValueError('template location is beyond the image boundaries!')

    bk_roi   = clutter_background[ly:ly + h_t, lx:lx + w_t]

    kcoef_global = np.sqrt(np.power(10, (snr / 10)) * erc / unk_numer)
    erc_local    = np.sum(bk_roi**2)/bk_roi.size
    kcoef_local  = np.sqrt(np.power(10, (snr / 10)) * erc_local / unk_numer)

    kcoef = kcoef_global
    if swerling_type == 0:          #swerling type 0 target
        kcoef_t = kcoef
        template = kuniform * kcoef_t
    if swerling_type == 1:          #central amplitude obeys the rayleigh distribution, which 2*sigma^2 = sigma_t = kcoef (swerling_0's Amplitude)
        sigma = kcoef#/np.sqrt(2)
        kcoef_t = rayleigh.rvs(loc=0, scale=sigma, size=1)
        template = kuniform * kcoef_t
    if swerling_type == 3:          #central amplitude obeys the chi distribution, which degrees of freedom k=4.
        kcoef_t =  chi2.rvs(df=kcoef, size=1) # or kcoef_t  = chi2.rvs(df=kcoef, size=1), then template=kgauss*kcoef
        template = kuniform*(kcoef_t)       # for chi2, Mean=df.
    loc_snr  = 10*np.log10(np.sum(template**2)/np.sum(bk_roi**2))
    glob_snr = 10*np.log10(np.sum(template ** 2)/(erc * template.size))
    # print('Swerling Type %d, kcoef_t %.2f (w %d, h %d), extened_unk %.2E' % (swerling_type, kcoef_t, w, h, unk_numer))
    # print('average (target - local clutter) power is (%.2f - %.2f)' % (np.sum(template ** 2) / template.size, erc_local))
    # print('Asked snr is %d, simulated local snr is %.2f, simulated global snr is %.2f' % (snr, loc_snr, glob_snr))
    local_snrs.append(loc_snr)
    global_snrs.append(glob_snr)
    #mask = ([template > bk_roi]) * template
    #clutter_background[ly:ly + h_t, lx:lx + w_t] = mask + bk_roi
    clutter_background[ly:ly + h_t, lx:lx + w_t] = template + bk_roi
    return clutter_background

def get_frame(img_w, img_h, frame_no, snr, gt_dict, swerling_type=0):
    '''
    Get one frame combine targets and clutter together.
    #add swerling type on Mar 2, 2021.
    :param frame_no:
    :return:
    '''
    frame_no_key = '%02d' % frame_no
    ray_background = rayleigh.rvs(loc=0, scale=1, size=(img_h, img_w)) #sigma_n=E(n^2) = 2*scale^2
    # Erc: average clutter energy.
    erc = np.sum(ray_background ** 2) / ray_background.size
    #add targets on the simulated position in each frame
    simulated_frame = ray_background
    # Each frame gets multiple targets.
    gt_targets =  gt_dict[frame_no_key]
    for tid in gt_targets:
        #Note that here x,y in gt is the top-lelf position.
        x, y, w, h, theta = gt_targets[tid]
        cx = x + w/2
        cy = y + h/2
        simulated_frame = add_gaussian_template_on_clutter_v2(cx, cy, w, h, theta, erc, snr,
                                                           simulated_frame,swerling_type)
        # if tid == 'amelia':#uniform distributed target.
        #     simulated_frame = add_uniform_template_on_clutter(cx, cy, w, h, theta, erc, snr, simulated_frame, swerling_type)
        # else:#Gaussian distributed target.
        #     simulated_frame = add_gaussian_template_on_clutter(cx, cy, w, h, theta, erc, snr, simulated_frame, swerling_type)
    #simulated_frame = uti.frame_normalize(simulated_frame)
    fids = list(gt_dict.keys())
    fids.sort()
    if(int(frame_no)==int(fids[-1])):
        print('Averaged (extended region -- peak point) SNR is (%.2f - %.2f)' % (np.mean(local_snrs), np.mean(global_snrs)))
    return simulated_frame

def manuver_in_clutter(snr=10):
    '''
    Simulate a target in a clutter given a snr.
    :return:
    '''
    img_w = 256
    img_h = 256

    rayscale = 1
    rayclutter = rayleigh.rvs(loc=0, scale=rayscale, size=(img_h, img_w))  # samples generation
    Erc = np.sum(rayclutter ** 2) / rayclutter.size

    xs,ys,ws,hs,thetas = s_manuver()

    for i, elem in enumerate(zip(xs,ys,ws,hs,thetas)):
        rayclutter = rayleigh.rvs(loc=0, scale=rayscale, size=(img_h, img_w))
        x, y, w, h, theta = elem
        et_clutter_frame = add_gaussian_template_on_clutter(x, y, w, h, theta, snr, rayclutter)
        plt.imshow(et_clutter_frame)
        plt.pause(0.1)

def multiple_extended_targets_in_clutter():
    '''
    :return:
    '''
    x0       = 20+20
    y0       = 30+20
    velo     = (1.5, 1.2)
    #velo    = (3.75, 2.7)
    npoints  = 51
    xs_cv, ys_cv = constant_velocity(x0, y0, velo, npoints)
    w_cv     = 20+8
    h_cv     = 16+4
    ws_cv    = np.ones(npoints)*w_cv #
    #ws_cv   = np.random.normal(w_cv, 0.5, npoints)
    hs_cv    = np.ones(npoints)*h_cv #
    #hs_cv   = np.random.normal(h_cv, 0.5, npoints)
    theta_cv = get_orientation(xs_cv, ys_cv)
    recttl_xs_cv = xs_cv - ws_cv/2
    recttl_ys_cv = ys_cv - hs_cv/2

    x0      = 160+20
    y0      = 30+20
    # velo  = (-6, -2)
    # acc   = (0.3, 0.25)
    velo    = (-1.5, -0.5)
    acc     = (0.1, 0.1)
    npoints = 51
    w_ca    = 28
    h_ca    = 20
    # w_ca  = 14 #for uniform_distribution
    # h_ca  = 20 #for uniform_distribution
    xs_ca, ys_ca = constant_accelerate(x0, y0, velo, acc, npoints)
    ws_ca    = np.ones(npoints)*w_ca ##
    #ws_ca   = np.random.normal(w_ca,  0.5, npoints)
    hs_ca    = np.ones(npoints)*h_ca ##
    #hs_ca   = np.random.normal(h_ca, 0.5, npoints)
    theta_ca = get_orientation(xs_ca, ys_ca)
    recttl_xs_ca = xs_ca - ws_ca/2
    recttl_ys_ca = ys_ca - hs_ca/2

    #radius = 60
    #omega = 0.0685
    # x0 = 50 + 6
    # y0 = 100+20
    radius  = 70
    omega   = 0.0685/1.5
    npoints = 51
    x0      = 50 + 6
    y0      = 100 + 20
    w_circ  = 16+20
    h_circ  = 10+10
    xs_circ, ys_circ = constant_turn(x0, y0, radius, omega, npoints)
    ws_circ = np.ones(npoints)*w_circ ##
    #ws_circ= np.random.normal(w_circ, 0.5, npoints)
    hs_circ = np.ones(npoints)*h_circ ##
    #hs_circ= np.random.normal(h_circ, 0.5, npoints)
    theta_circ = get_orientation(xs_circ, ys_circ)
    recttl_xs_circ = xs_circ - ws_circ/2
    recttl_ys_circ = ys_circ - hs_circ/2

    # radius = 50
    # omega = -0.15
    # npoints = 50
    # x0 = 60 + 20
    # y0 = 100 + 20
    # w_ct = 16+10
    # h_ct = 16+0
    # xs_ct, ys_ct = constant_turn(x0, y0, radius, omega, npoints)
    # ws_ct = np.random.normal(w_ct, 0.5, npoints)
    # hs_ct = np.random.normal(h_ct, 0.5, npoints)
    # theta_ct = get_orientation(xs_ct, ys_ct)

    # x0 = 40
    # y0 = 30+20
    # velo = (0.5, 0)
    # npoints = 50
    # w_cvline = 22 + 16
    # h_cvline = 17 + 13
    # xs_cvline, ys_cvline = constant_velocity(x0, y0, velo, npoints)
    # #ws_ca = np.ones(npoints)*w_ca ##
    # ws_cvline = np.random.normal(w_cvline,  0.5, npoints)
    # #hs_ca = np.ones(npoints)*h_ca ##
    # hs_cvline  = np.random.normal(h_cvline, 0.5, npoints)
    # theta_cvline = get_orientation(xs_cvline, ys_cvline)
    # recttl_xs_cvline = xs_cvline - ws_cvline/2
    # recttl_ys_cvline = ys_cvline - hs_cvline/2

    ## This part is to view the trajectory of the ideal ground-truth.
    # fig,ax =plt.subplots()
    # plot_ellipse(ax, xs_cv,   ys_cv, ws_cv, hs_cv, facecolor='green')
    # plot_ellipse(ax, xs_ca,   ys_ca, ws_ca, hs_ca, facecolor='red')
    # plot_ellipse(ax, xs_circ, ys_circ, ws_circ, hs_circ, facecolor='blue')
    # plot_ellipse(ax, xs_ct,   ys_ct, ws_ct, hs_ct, facecolor='black')
    # plt.show()

    Gt_dict = {}
    for i in range(npoints):
        Gt_dict['%02d' % i] = {}
        # tid = 1, 2, 3, 4
        Gt_dict['%02d' % i]['victor']=[recttl_xs_cv[i],  recttl_ys_cv[i],  ws_cv[i],  hs_cv[i], theta_cv[i]]
        Gt_dict['%02d' % i]['amelia']=[recttl_xs_ca[i],  recttl_ys_ca[i],  ws_ca[i],  hs_ca[i], theta_ca[i]]
        Gt_dict['%02d' % i]['urich' ]=[recttl_xs_circ[i],recttl_ys_circ[i],ws_circ[i], hs_circ[i], theta_circ[i]]
        #Gt_dict['%02d' % i]['line' ] =[recttl_xs_cvline[i], recttl_ys_cvline[i], ws_cvline[i], hs_cvline[i], theta_cvline[i]]
        #Gt_dict['%02d' % i]['dormy']=[xs_ct[i],  ys_ct[i],  ws_ct[i],  hs_ct[i], theta_ct[i]]

    # # add target on the clutter background
    # # results can be viewed on a canvas(300,300).
    # img_w = 300
    # img_h = 300
    #
    # rayscale = 1  # Base uint for computing the snr.
    # rayclutter = rayleigh.rvs(loc=0, scale=rayscale, size=(img_h, img_w))  # samples generation
    # Erc = np.sum(rayclutter ** 2) / rayclutter.size
    #
    # snr = 10
    # frame_nums = len(Gt_dict)
    # for key in Gt_dict:
    #     print('frame %s' % key)
    #     gt_targets = Gt_dict[key]
    #     for tid in gt_targets:
    #         x, y, w, h, theta = gt_targets[tid]
    #         et_clutter_frame  = add_gaussian_template_on_clutter(x, y, w, h, theta, snr, rayclutter)
    #         plt.imshow(et_clutter_frame)
    #         plt.pause(0.1)
    return Gt_dict


def mtt_sim():
    '''
    simulate 4 targets in a roi to test the JPDA algorithm.
    :return:
    '''
    x0 = 10
    y0 = 20
    velo = (1.5, 1.7)
    npoints = 50
    x1s, y1s = constant_velocity(x0, y0, velo, npoints)

    x0 = 10
    y0 = 80
    velo = (1.5, -2)
    npoints = 50
    x2s, y2s = constant_velocity(x0, y0, velo, npoints)

    radius = 60
    omega =  0.0685
    npoints =50
    x0 =  30
    y0 =  50
    x3s,y3s = constant_turn(x0, y0, radius, omega, npoints)

    radius = 50
    omega =  -0.15
    npoints =50
    x0 =  60
    y0 =  100
    x4s,y4s = constant_turn(x0, y0, radius, omega, npoints)

    plt.axis([0, 200, 0, 200])
    plt.plot(x1s, y1s, '.', color='red')
    plt.plot(x2s, y2s, '.', color='green')
    plt.plot(x3s, y3s, '.', color='blue')
    plt.plot(x4s, y4s, '.', color='yellow')
    tx = [str(i) for i in range(1,51)]

    # x = x1s
    # y = y1s
    # for i in range(50):
    #     plt.text(x[i], y[i], tx[i])
    #plt.text(x1s, y1s, tx)
    show_text(x1s, y1s, tx)
    show_text(x2s, y2s, tx)
    show_text(x3s, y3s, tx)
    show_text(x4s, y4s, tx)
    plt.show()

def plot_ellipse(ax, xs, ys, ws, hs, facecolor):
    '''
    Plot ellipse based on the ground truth sequential points:
    :param ax: axis object
    :param xs: x vector
    :param ys: y vector
    :param ws: width vector
    :param hs: height vector
    :return:
    '''
    dys = np.diff(ys)
    dxs = np.diff(xs)
    thetas_less = np.arctan2(dys, dxs) # len(dxs) - 1
    thetas = np.pad(thetas_less,(0,1),'edge') # add one elements to the end
    #ellipse_gate1 = []
    #fig, ax = plt.subplots()
    #plot_trajectory(xs, ys, color=facecolor)
    for i in range(len(xs)):
        #rect = Rectangle(xy=[x1s[i], y1s[i]], width=w1s[i], height=y1s[i], angle=theta1s[i])
        angle_deg = thetas[i]*180/np.pi
        e = Ellipse(xy=[xs[i], ys[i]], width=ws[i], height=hs[i], angle=angle_deg, alpha=0.5, color=facecolor)
        #ellipse_gate1.append(e)
        plt.plot(xs, ys, '.', color=facecolor, markersize=2)
        ax.add_patch(e)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.text(xs[i], ys[i], str(i), fontsize=9, color=facecolor)




def multiple_extended_targets_sim():
    '''
    simulate 4 extended targets in a roi, pay attention to rotation.
    theta = atan(dy/dx)
    :return:
    '''
    x0 = 10
    y0 = 20
    #velo = (1.5, 1.7)
    velo = (1.5, 2.7)
    npoints = 50
    x1m, y1m = constant_velocity(x0, y0, velo, npoints)

    motion_noise      = np.random.normal(3,0.4,2*npoints)
    observation_noise = np.random.normal(2,0.5,2*npoints)
    x1t = x1m + motion_noise[0:npoints]
    y1t = y1m + motion_noise[npoints:2*npoints]
    w1t = 4
    h1t = 2
    x1s = x1t + observation_noise[:npoints]
    y1s = y1t + observation_noise[npoints:2*npoints]
    w1s = np.random.normal(w1t, 0.5, npoints)
    h1s = np.random.normal(h1t, 0.5, npoints)

    x0 = 10
    y0 = 80
    velo = (1.5, -2)
    npoints = 50
    x2m, y2m = constant_velocity(x0, y0, velo, npoints)

    motion_noise      = np.random.normal(4,0.5,2*npoints)
    observation_noise = np.random.normal(2,0.5,2*npoints)
    x2t = x2m + motion_noise[0:npoints]
    y2t = y2m + motion_noise[npoints:2*npoints]
    w2t = 4
    h2t = 3
    x2s = x2t + observation_noise[:npoints]
    y2s = y2t + observation_noise[npoints:2*npoints]
    w2s = np.random.normal(w2t, 0.5, npoints)
    h2s = np.random.normal(h2t, 0.5, npoints)

    radius = 60
    omega =  0.0685
    npoints =50
    x0 =  30
    y0 =  50
    x3m, y3m = constant_turn(x0, y0, radius, omega, npoints)
    motion_noise      = np.random.normal(3,0.5,2*npoints)
    observation_noise = np.random.normal(2,0.5,2*npoints)
    x3t = x3m + motion_noise[0:npoints]
    y3t = y3m + motion_noise[npoints:2*npoints]
    w3t = 6
    h3t = 3
    x3s = x3t + observation_noise[:npoints]
    y3s = y3t + observation_noise[npoints:2*npoints]
    w3s = np.random.normal(w3t, 0.5, npoints)
    h3s = np.random.normal(h3t, 0.5, npoints)

    radius = 50
    omega =  -0.15
    npoints =50
    x0 =  60
    y0 =  100
    x4m,y4m = constant_turn(x0, y0, radius, omega, npoints)
    motion_noise      = np.random.normal(3,0.5,2*npoints)
    observation_noise = np.random.normal(2,0.5,2*npoints)
    x4t = x4m + motion_noise[0:npoints]
    y4t = y4m + motion_noise[npoints:2*npoints]
    w4t = 5
    h4t = 2
    x4s = x4t + observation_noise[:npoints]
    y4s = y4t + observation_noise[npoints:2*npoints]
    w4s = np.random.normal(w4t, 0.5, npoints)
    h4s = np.random.normal(h4t, 0.5, npoints)
    Zs_dict = {}
    Xt_dict = {}
    for i in range(npoints):
        Zs_dict['%d' % i] = []
        Zs_dict['%d' % i].append(np.array([ [x1s[i]], [y1s[i]], [w1s[i]], [h1s[i]] ]))
        Zs_dict['%d' % i].append(np.array([ [x2s[i]], [y2s[i]], [w2s[i]], [h2s[i]] ]))
        Zs_dict['%d' % i].append(np.array([ [x3s[i]], [y3s[i]], [w3s[i]], [h3s[i]] ]))
        Zs_dict['%d' % i].append(np.array([ [x4s[i]], [y4s[i]], [w4s[i]], [h4s[i]] ]))
        Xt_dict['%d' % i] = []
        Xt_dict['%d' % i].append(np.array([ [x1t[i]], [y1t[i]], [w1t], [h1t] ]))
        Xt_dict['%d' % i].append(np.array([ [x2t[i]], [y2t[i]], [w2t], [h2t] ]))
        Xt_dict['%d' % i].append(np.array([ [x3t[i]], [y3t[i]], [w3t], [h3t] ]))
        Xt_dict['%d' % i].append(np.array([ [x4t[i]], [y4t[i]], [w4t], [h4t] ]))
    # plt.axis([0, 200, 0, 200])
    # plt.plot(x1s, y1s, '.', color='red')
    # plt.plot(x2s, y2s, '.', color='green')
    # plt.plot(x3s, y3s, '.', color='blue')
    # plt.plot(x4s, y4s, '.', color='yellow')
    # tx = [str(i) for i in range(1,51)]
    # show_text(x1s, y1s, tx)
    # show_text(x2s, y2s, tx)
    # show_text(x3s, y3s, tx)
    # show_text(x4s, y4s, tx)
    # plt.show()
    fig, ax = plt.subplots()
    plot_ellipse(ax, x1s, y1s, w1s, h1s, facecolor='red')
    plot_ellipse(ax, x2s, y2s, w2s, h2s, facecolor='green')
    plot_ellipse(ax, x3s, y3s, w3s, h3s, facecolor='blue')
    plot_ellipse(ax, x4s, y4s, w4s, h4s, facecolor='black')
    plt.show()
    return Zs_dict, Xt_dict



def show_text(xs, ys, tx):
    num = len(xs)
    for i in range(num):
        plt.text(xs[i], ys[i], tx[i])


def plot_trajectory(xs, ys, color='red'):
    '''
    Draw the trajectory on the whiteboard.
    :param xs:
    :param ys:
    :return:
    '''
    plt.plot(xs, ys, '.', color=color)


def test_kf(xs, ys, mx, my):
    '''
    :param xs: ground truth x
    :param ys: ground truth y
    :param mx: measured x
    :param my: measured y
    :return:
    '''

    tracker = kf_model.kalman_filter()
    cx = xs[0]
    cy = ys[0]
    ok = tracker.init(cx, cy)
    X_ = tracker.X0
    P_ = tracker.P0

    ex = [cx]
    ey = [cy]

    N = len(xs)-1
    for i in range(1, N):
        zx = mx[i]
        zy = my[i]
        X_, P_, Xpre = tracker.update(X_, P_, zx, zy)
        ex.append(X_[0,0])
        ey.append(X_[2,0])

    plot_trajectory(xs, ys, 'red')
    plot_trajectory(mx, my, 'yellow')
    plot_trajectory(ex, ey, 'green')

def get_cov_ellipse(mean, cov):
    '''
    Get ellipse from a 2d Gaussian covariance.
    :param mean:
    :param cov:
    :return:
    '''
    w, v      = la.eig(cov)
    angle_deg = np.arctan2(v[1, 0], v[0, 0])
    angle_deg *= 180./np.pi
    e = Ellipse(xy=mean, width=w[0], height=w[1], angle=angle_deg, alpha=0.5, color='black')
    return e

def test_ettkf(xs, ys, mx, my, mw, mh):
    '''
    :param xs: ground truth x
    :param ys: ground truth y
    :param mx: measured x
    :param my: measured y
    :return:
    '''
    tracker = kf_model.ETT_KF_Filter()
    cx = xs[0]
    cy = ys[0]
    vx = 4
    vy = 4
    w  = 3
    h  = 1.5
    ok = tracker.init(cx, vx,  cy,  vy, w, h)
    X_ = tracker.X0
    P_ = tracker.P0

    ex = [cx]
    ey = [cy]
    ellipse_gate = []
    ett_rects = []

    zxpr = []
    zypr = []

    N = len(xs)
    gamma_gate = 5
    for i in range(1, N):
        zx = mx[i]
        zy = my[i]
        zw = mw[i]
        zh = mh[i]
        X_, P_, X_pr, Z_pr, S, S_inv = tracker.update(X_, P_, zx, zy, zw, zh)
        ex.append(X_[0,0])
        ey.append(X_[2,0])
        zxpr.append(Z_pr[0,0])
        zypr.append(Z_pr[1,0])
        # Only get the x,y mean and cov for ellipse fitting
        eli  = get_cov_ellipse(Z_pr[0:2], S[0:2,0:2])
        rect = Rectangle(xy=[X_[0,0]-X_[4,0]/2, X_[2,0]-X_[5,0]/2], width=zw, height=zh,angle=eli.angle)
        ellipse_gate.append(eli)
        ett_rects.append(rect)

    fig, ax = plt.subplots()
    plot_trajectory(xs, ys, 'red')
    plot_trajectory(mx, my, 'yellow')
    plot_trajectory(ex, ey, 'green')
    plot_trajectory(zxpr, zypr, 'blue')
    for eg in ellipse_gate:
        ax.add_artist(eg)
    for rect in ett_rects:
        ax.add_artist(rect)
    plt.show()

def test_jpda():
    '''
    Test JPDA.
    :return:
    '''
    mtt_jpda = jpda_model.Traj_manage()
    mtt_jpda.init()
    Zs_dict, Xt_dict = multiple_extended_targets_sim()
    nframe = 0
    for key in Zs_dict:
        print('frame %s' % key)
        Zs = Zs_dict[key]
        if nframe == 0 :
            mtt_jpda.track_init(Zs)
        else:
            mtt_jpda.track_update(Zs)
        nframe += 1
        print('')

def test_distribution_average_power():
    '''
    Test the average power (ap, or mean squared amplitude) of random point which are sampled from a known distribution.
    Monitoring the relationship between the sigma and the parameters of the distribution.
    :return:
    '''
    print('Test for point samples.')
    ray_sigma = 2
    ray_scale = np.sqrt(ray_sigma/2)
    ray_samples = rayleigh.rvs(loc=0, scale=ray_scale, size=10000)
    num_ap_ray = np.mean(ray_samples**2)

    sea_clutter_samples   = rayleigh.rvs(loc=0, scale=ray_scale, size=10000)
    target_add_sc_samples = ray_samples+sea_clutter_samples
    test_snr = 10*np.log10(np.mean((target_add_sc_samples - np.sqrt(2))**2)/2)
    print('Rayleigh theroy average power (ap) is %.2f, numerical ap is %.2f'%(ray_sigma, num_ap_ray))
    print('sw=1 target in clutter, snr %.2f'%test_snr)

    chi_sigma = 2
    chi_df    = 2
    chi_samples = chi.rvs(df=chi_df, loc=0, size=10000)
    num_ap_chi = np.mean(chi_samples**2)
    print('Chi      theroy average power (ap) is %.2f, numerical ap is %.2f'%(chi_sigma, num_ap_chi))

    chi2_sigma = 2
    # Reversely eq: 2*df+df^2 = sigma_t
    chi2_df    = np.sqrt(chi2_sigma+1)-1
    #scale = np.sqrt(E(x^2)/(2*df+df^2))
    chi2_samples = chi2.rvs(df=4,  size=10000, scale=1/np.sqrt(12))
    num_ap_chi2  = np.mean(chi2_samples**2)
    print('Chi2     theroy average power (ap) is %.2f, numerical ap is %.2f'%(chi2_sigma, num_ap_chi2))


    print('Test for extended target samples.')

    w = 28
    h = 20
    theta=45
    sigma_x = (w / 2.5 - 0.5) / 2 #sigma_x is related to the width of the template
    sigma_y = (h / 2.5 - 0.5) / 2

    kgauss = gaussian_kernel2d(sigma_x, sigma_y, theta, bnorm=False)  # Get diffusive coefficients for a 2d gaussian
    Egk_numer = np.sum(kgauss.ravel() ** 2) / kgauss.size  # 2d gaussian's average power.

    h_t, w_t = kgauss.shape

    snr = 0
    erc = 1
    # compute the amplitude coefficients according to the SNR Eq.
    sigma_t = np.power(10, (snr / 10)) * erc/Egk_numer

    # rayleigh_scale = np.sqrt(sigma_t / 2)
    # ray_frame_samples = rayleigh.rvs(loc=0, scale=rayleigh_scale, size=10000)
    # df = 4
    # chi2_scale = np.sqrt(sigma_t / (2 * df + df ^ 2))
    # chi2_frame_samples= chi2.rvs(df=df, scale=chi2_scale, size=10000)
    # plt.hist(ray_frame_samples, color='r', alpha=0.5, bins=range(12))
    # #plt.figure()
    # plt.hist(chi2_frame_samples,color='y', alpha=0.5, bins=range(12))
    # plt.pause(0.1)

    # average power of clutter is computed by numerical results in local roi-window.
    num_snr_list = []
    swerling_type = 3
    for i in range(1000):
        if swerling_type == 0:  # swerling type 0 target
            a = np.sqrt(sigma_t)
            template = kgauss * a
        if swerling_type == 1:
            rayleigh_scale = np.sqrt(sigma_t/2)
            # central amplitude obeys the rayleigh distribution, which 2*sigma^2 = sigma_t = kcoef**2 (swerling_0's Amplitude)
            a = rayleigh.rvs(loc=0, scale=rayleigh_scale, size=1)
            #a =  np.mean(a_rvs)
            template = kgauss * a
        if swerling_type == 3:  # central amplitude obeys the chi distribution, which degrees of freedom k=4.
            #df= np.sqrt(sigma_t+1)-1
            df= 4
            chi2_scale = np.sqrt(sigma_t/(2*df+df^2))
            a = chi2.rvs(df=df, size=1, scale=chi2_scale)  # or kcoef_t  = chi2.rvs(df=kcoef, size=1), then template=kgauss*kcoef
            #a     = np.mean(a_rvs)
            template = kgauss * a  # for chi2, Mean=df.
        num_snr = 10*np.log10(np.mean(template**2)/erc)
        num_snr_list.append(num_snr)
    print('swerling %d, numerical snr is %.5f'%(swerling_type, np.average(num_snr_list)))
    print()


if __name__ == '__main__':
    test_distribution_average_power()
    #manuver_in_clutter()

    # from PIL import Image
    #
    # data = np.random.random((2, 2))
    # img1 = Image.fromarray(data)
    # img1.save('test.tiff',dpi=(300,300))
    # img2 = Image.open('test.tiff')
    #
    # f1 = np.array(img1.getdata())
    # f2 = np.array(img2.getdata())
    # print(f1==f2)
    # print(f1)


    multiple_extended_targets_in_clutter()

    #using simulated targets to test jpda algorithm
    #test_jpda()
    #
    # mean = [19.92977907, 5.07380955]
    # width = 30
    # height = 10.1828848
    # angle = 0
    # ell = Ellipse(xy=mean, width=width, height=height, angle=angle,fill=None)
    # fig, ax = plt.subplots()
    #
    # ax.add_patch(ell)
    # ax.set_aspect('equal')
    # ax.autoscale()
    # plt.show()

    #multiple_extended_targets_sim()
    xs,ys,ws,hs,thetas = s_manuver()
    print('')

    # mtt_sim()
    # plt.show()
    #plt.show()
    #npts = 100
    # x0 = 0
    # y0 = 0
    # velo=(4,4)
    # xs,ys = constant_velocity(x0,y0,velo,npts)

    # x0=0
    # y0=200
    # velo=(2,2)
    # acc = 0.01
    # xs,ys =  constant_accelerate(x0,y0,velo,acc,npts)

    # x0=50
    # y0=50
    # radius = 50
    # omega = 0.2
    # xs,ys = constant_turn(x0, y0, radius, omega, npts)



    # mmean = [0,0]
    # mcov  = [[2,  0],
    #          [0,2.5]]
    #
    # dx, dy = np.random.multivariate_normal(mmean, mcov, npts).T
    # mx     = xs+dx
    # my     = ys+dy
    # #gaussian_disturbance = norm.rvs(loc=0, scale=1, size=(1, npts))
    # # plot_trajectory(xs,ys,'red')
    # # plot_trajectory(mx,my,'yellow')
    # #test_kf(xs,ys,mx,my)
    #
    # w = 3
    # h = 1.5
    # mw = w + dx
    # mh = h + dy
    # test_ettkf(xs, ys, mx, my, mw, mh)

    # x0=0
    # y0=200
    # velo=2
    # acc = 0.01
    # xs,ys =  constant_accelerate(x0,y0,velo,acc,npts)
    # plot_trajectory(xs,ys,'blue')
    #
    # x0=50
    # y0=50
    # radius = 50
    # omega = 0.2
    # xs,ys = constant_turn(x0, y0, radius, omega, npts)
    # plot_trajectory(xs,ys,'green')
    # plt.show()

