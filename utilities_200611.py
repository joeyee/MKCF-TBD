'''
utilities for auto segmenting and tracking in inesa_2018 datasets
Created 20200611@Provence_Dalian
'''
import cv2
import glob
import os
from PIL import Image                      # saving image with high dpi
import numpy as np
import matplotlib.pyplot as plt

def frame_normalize(frame):
    '''
    :param frame: source frame
    :return: normalized frame in double float form
    '''
    # 归一化操作
    # dframe is for computing in float32
    dframe = frame.astype(np.float32) #for the convenient of opencv operation
    #normalization
    dframe = (dframe - dframe.min()) / (dframe.max() - dframe.min())
    return dframe

def waitkey_imshow(canvas, frame_no):
    plt.imshow(canvas)
    plt.title('frame %02d' % frame_no)
    plt.draw()
    plt.pause(0.0001)
    key_press = False
    while not key_press:
        key_press = plt.waitforbuttonpress()

def draw_rect(img, rect, color = (0,255,0), thick = 1):
    '''
    draw a rectangle on img, for convinient using than cv2
    :param img:
    :param rect:
    :return: img
    '''
    p1 = (int(rect[0]), int(rect[1]))
    p2 = (int(rect[0]+rect[2]), int(rect[1]+rect[3]))
    img = cv2.rectangle(img, p1, p2, color, thick)
    return img

def get_subwindow(im, pos, sz):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.
    """
    if np.isscalar(sz):  # square sub-window
        sz = [sz, sz]

    ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0]/2)
    xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1]/2)

    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates,
    # and set them to the values at the borders
    ys[ys < 0] = 0
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    # extract image
    template = im[np.ix_(ys, xs)]
    return template

def intersection_rect(recta, rectb):
    '''
    Intersection area of two rectangles.
    :param recta:
    :param rectb:
    :return: iou rate.
    '''
    tlx = max(recta[0], rectb[0])
    tly = max(recta[1], rectb[1])
    brx = min(recta[0]+recta[2], rectb[0]+rectb[2])
    bry = min(recta[1]+recta[3], rectb[1]+rectb[3])

    intersect_area = max(0., brx-tlx+1) * max(0., bry-tly+1)
    #intersect_area = max(0., brx - tlx) * max(0., bry - tly)
    iou = intersect_area/(recta[2]*recta[3] + rectb[2]*rectb[3] - intersect_area + np.spacing(1))
    iou = min(1, iou)
    return iou

def intersection_area(recta, rectb):
    '''
    Intersection area of two rectangles.
    :param recta:
    :param rectb:
    :return: iou area.
    '''
    tlx = max(recta[0], rectb[0])
    tly = max(recta[1], rectb[1])
    brx = min(recta[0]+recta[2], rectb[0]+rectb[2])
    bry = min(recta[1]+recta[3], rectb[1]+rectb[3])
    intersect_area = max(0., brx-tlx) * max(0., bry-tly)
    return intersect_area

def save_gray2jet():
    '''
    Save gray image in the defined path to jet images in ./Jet
    :param frame:
    :param path:
    :return:
    '''
    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    save_path = file_prefix + '/Jet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_names = glob.glob(file_prefix + '*.png')
    file_names.sort()
    file_len = len(file_names)


    for i in range(0, file_len):
        fname_split = file_names[i].split('/')
        frame_no = int(fname_split[-1].split('.')[0])
        print('frame no %d' % frame_no)
        frame = cv2.imread(file_names[i], 0)
        canvas = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # for plt.imshow()
        save_frame = Image.fromarray(canvas)
        fname = '%02d.png'%frame_no
        save_frame.save(save_path+fname, dpi=(72, 72), compress_level=0)

def save_frame(frame, fid):
    fig = plt.figure(figsize=(20, 10.16), facecolor='white', dpi=300)
    ax  = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1) #no white margin left
    ax.imshow(frame)
    # ax.set_xticks([])
    # ax.set_yticks([])
    if fid ==42:
        profile = frame[42,:]
        f,a = plt.subplots()
        a.plot(profile, color='blue',linewidth=1.5, alpha=0.8)

    f.savefig('/Users/yizhou/code/taes2021/results/k_distributed_frames/frame%d_profile.png'%fid, dpi=300, bbox_inches='tight')
    plt.close(f)

    path = '/Users/yizhou/code/taes2021/results/k_distributed_frames/'
    sframe = Image.fromarray(frame)
    sframe.save('%s/%d.png' % (path, fid), compress_level=0, dpi=(300,300))

if __name__=='__main__':
    save_gray2jet()