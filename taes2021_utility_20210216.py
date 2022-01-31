'''
This file contains functions which support TAES2021 trackers.
Including load ground truth information.
Created by Yi ZHOU@Provence_Dalian on 0216-2021.
Based on files 'MCF_TBD_Param_Test_Titan_20210205.py'
'''
import json         #For labelme's groundtruth.
import cv2
import numpy                      as np
import utilities_200611           as uti            # personal tools
import matplotlib.pyplot          as plt            # observing plots
import matplotlib

#Get the Gt information from the JSON file.
def get_gt_rect(json_file_name):
    '''
    From the json_file gets the ground truth rectangles and nick name for each target.
    :param json_file_name:
    :return: target_dict
    '''
    target_dict ={} # [nick_name]:Rect
    fname_split = json_file_name.split('/')
    frame_no    = int(fname_split[-1].split('.')[0])
    #print('frame no %d' % frame_no)
    with open(json_file_name) as f:
        data = json.load(f)
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if(label_name == 'Sherry'):
                continue
            points     = shape["points"] #vertex of the polygon.
            x, y, w, h = cv2.boundingRect(np.array([np.int0(points)]))
            target_dict[label_name] = [x,y,w,h]
    return target_dict

def get_gt_region(json_file_name):
    '''
    From the json_file gets the ground truth polygon and nick name for each target.
    :param json_file_name:
    :return: target_region_dict
    '''
    target_region_dict ={} # [nick_name]:Rect
    fname_split = json_file_name.split('/')
    frame_no    = int(fname_split[-1].split('.')[0])
    #print('frame no %d' % frame_no)
    with open(json_file_name) as f:
        data = json.load(f)
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if(label_name == 'Sherry'):
                continue
            points     = shape["points"]
            x, y, w, h = cv2.boundingRect(np.array([np.int0(points)]))
            #target_region_dict[label_name] = [x,y,w,h]
            target_region_dict[label_name] = points
    return target_region_dict

def convert_rotateRect_to_Rect_trajectory_dict(rotate_rect_dict):
    '''
    convert the rotate rect trjaectory from gt file to bounding rectangle trajectory
    :param rotate_rect_dict: {fid:{tname:[x,y,w,h,theta]}, fid2:{}}
    :param rect_dict: {fid:{tname:[bx,by,bw,bh]}, fid2:{}}
    :return:
    '''
    rect_trajectory_dict = {}
    for fid in rotate_rect_dict:
        rect_trajectory_dict[int(fid)] = {}
        for tname in rotate_rect_dict[fid]:
            x, y, w, h, theta = rotate_rect_dict[fid][tname]
            bounding_box_of_ellipse = ((x + w / 2, y + h / 2), (w, h), theta * 180 / np.pi)
            rect_vertex   = cv2.boxPoints(bounding_box_of_ellipse)
            bounding_rect = cv2.boundingRect(np.int0(rect_vertex))
            #shrink the bounding_rect 2 times.
            # x, y, w, h = bounding_rect[:4]
            # cx, cy = [int(x + w / 2), int(y + h / 2)]
            # gt_shrink_box = [int(cx - w / 4), int(cy - h / 4), int(w / 2), int(h / 2)]
            # rect_trajectory_dict[int(fid)][tname] = gt_shrink_box
            rect_trajectory_dict[int(fid)][tname] = bounding_rect
    return rect_trajectory_dict

def get_pfa_pd_via_trajectory_rrect(trk_traj, gt_traj, frame_height, frame_width, bshrink_tbox=True):
    '''
    This function is used for the simulated ground_truth, with shrink Gt and trk for the Gaussian extended target.
    Get prob. of false alarm and prob. of detection from tracker's trajectory.
    trk_traj{fid1:{tid1:[x,y,w,h], tid2:[x2,y2,w2,h2],...}, fid2:{tid1:[x,y,w,h], ....}, ....}
    gt_traj{fid1:{tname1:[x,y,w,h], tname2:[]...}, fid2:{tname1:[x,y,w,h]...},...}
    :param trk_traj:
    :param gt_traj:
    :return:
    '''
    gt_mask        = np.zeros((frame_height, frame_width)) # gt_mask  should set all the pixels  in gt_rects  as 1
    gt_shrink_mask = np.zeros((frame_height, frame_width)) # shrink gt extended target to increase pd(same as pd_seg computing)
    trk_mask       = np.zeros((frame_height, frame_width)) # trk_mask should set all the pixels  in trk_rects as 1
    fid_list= list(gt_traj.keys())
    pfa_list= []
    pd_list = []
    for fid in gt_traj:
        gt_mask  = gt_mask *0        #clear gt  mask in each frame.
        trk_mask = trk_mask*0        #clear trk mask
        gt_shrink_mask = gt_shrink_mask*0
        for tname in  gt_traj[fid]:
            rect = gt_traj[fid][tname]
            x, y, w, h = rect
            gt_mask[y:y + h, x:x + w] = 1  # set the pixels in  gt rects as 1.
            #shrink the gt rect.
            cx, cy = [int(x + w / 2), int(y + h / 2)]
            w = int(0.55 * w)
            h = int(0.55 * h)
            sx, sy = [int(cx - w / 2), int(cy - h / 2)]
            shrink_gt_rect = [sx, sy, w, h]
            # Shrink the gt box will increase the detection rate for clustering Gaussian Extended target.
            gt_shrink_mask[sy:sy + h, sx:sx + w] = 1
        if fid in trk_traj :
            for tid in trk_traj[fid]:
                trk_rect = trk_traj[fid][tid]
                tx, ty, tw, th = trk_rect
                if bshrink_tbox: #shrink tbox or not, this is true for MCF, false for Grossi.
                    #MCF tracker exlarge the tbox for avoiding the frequency domain aliasing.
                    cx, cy = [int(tx + tw / 2), int(ty + th / 2)]
                    # Shrink the trk_box for MCF will decrease the false alarm rate for clustering Gaussian Extended target.
                    tw, th = [int(tw*3/4), int(th*3/4)]
                    tx,ty = [int(cx - tw / 2), int(cy - th / 2)]
                trk_mask[ty:ty + th, tx:tx + tw] = 1  # set the pixels in  track rects as 1.

        pfa = (np.sum(trk_mask) - np.sum(trk_mask*gt_mask))/(frame_height*frame_width - np.sum(gt_mask))
        pd  = np.sum(trk_mask*gt_shrink_mask) / np.sum(gt_shrink_mask)
        pfa_list.append(pfa)
        pd_list.append(pd)
    return fid_list, pfa_list, pd_list

def get_pfa_pd_via_trajectory_rrect_v2(trk_traj, gt_traj, precision_dict, frame_height, frame_width, bshrink_tbox=True):
    '''
    This function is used for the simulated ground_truth, with shrink Gt and trk for the Gaussian extended target.
    Get prob. of false alarm and prob. of detection from tracker's trajectory.
    trk_traj{fid1:{tid1:[x,y,w,h], tid2:[x2,y2,w2,h2],...}, fid2:{tid1:[x,y,w,h], ....}, ....}
    gt_traj{fid1:{tname1:[x,y,w,h], tname2:[]...}, fid2:{tname1:[x,y,w,h]...},...}

    Note:v2 version modify the mismatch detection problem. If a mismatch happens for two true target, the tracker's detection
    rate should be dropped. precision_dict gives gt_target and associated tracker'tid. precision[tname_gt][tid_tracker]
    :param trk_traj:
    :param gt_traj:
    :return:
    '''
    gt_mask = np.zeros((frame_height, frame_width))         # gt_mask  should set all the pixels  in gt_rects  as 1
    gt_shrink_mask = np.zeros((frame_height, frame_width))  # shrink gt extended target to increase pd(same as pd_seg computing)
    trk_mask = np.zeros((frame_height, frame_width))        # trk_mask should set all the pixels  in trk_rects as 1
    fid_list = list(gt_traj.keys())
    pfa_list = []
    pd_list  = []
    for fid in gt_traj:
        gt_mask  = gt_mask * 0    # clear gt  mask in each frame.
        trk_mask = trk_mask * 0  # clear trk mask
        gt_shrink_mask = gt_shrink_mask * 0
        tp_pixels      = 0 # True positive in the rect of the trajectory tail.
        fp_pixels      = 0 # False positive in the trajectory per frame.
        for tname in gt_traj[fid]:
            gt_box = gt_traj[fid][tname]
            x, y, w, h = gt_box
            gt_mask[y:y + h, x:x + w] = 1  # set the pixels in  gt rects as 1.
            # shrink the gt rect.
            cx, cy = [int(x + w / 2), int(y + h / 2)]
            w = int(0.75*w)
            h = int(0.75*h)
            sx,sy = [int(cx-w/2), int(cy-h/2)]
            shrink_gt_rect = [sx,sy, w, h]
            # Shrink the gt box will increase the detection rate for clustering Gaussian Extended target.
            gt_shrink_mask[sy:sy+h, sx:sx+w] = 1
            matched_tids = precision_dict[tname]

            b_gt_detected = False
            if fid in trk_traj:
                for tid in trk_traj[fid]:
                    if tid in matched_tids:  # a match happens
                        tp_trk_rect = trk_traj[fid][tid]
                        iou = uti.intersection_rect(tp_trk_rect, gt_box)
                        if iou>0.15:
                            b_gt_detected = True #gt_box is detected for sure.
                    else: # unmatched rects are false positive.
                        fp_trk_rect = trk_traj[fid][tid]
                        tx, ty, tw, th = fp_trk_rect[:4]
                        if bshrink_tbox == True:
                            #shrink tbox or not, this is true for MCF, false for Grossi.
                            #Since the initialized new KCF tracker in MCF and enlarger the width and height twice times.
                            # MCF tracker exlarge the tbox for avoiding the frequency domain aliasing.
                            tcx, tcy = [int(tx + tw / 2), int(ty + th / 2)]
                            # Shrink the trk_box for MCF will decrease the false alarm rate for clustering Gaussian Extended target.
                            tw, th = [int(tw * 1 / 2), int(th * 1 / 2)]
                            tx, ty = [int(tcx - tw / 2), int(tcy - th / 2)]
                            trk_mask[ty:ty + th, tx:tx + tw] = 1  # set the pixels in  track rects as 1.
                            #fp_pixels  += fp_trk_rect[2]*fp_trk_rect[3]*((0.25)**2)
                        else:
                            trk_mask[ty:ty + th, tx:tx + tw] = 1
                            #fp_pixels  += fp_trk_rect[2]*fp_trk_rect[3]
            if b_gt_detected:
                tp_pixels += gt_box[2] * gt_box[3]
        fp_pixels = np.sum(trk_mask)
        pfa = (fp_pixels) / (frame_height * frame_width - np.sum(gt_mask))
        pd  =  min(tp_pixels, np.sum(gt_mask))/np.sum(gt_mask)
        pfa_list.append(pfa)
        pd_list.append(pd)
    return fid_list, pfa_list, pd_list

def get_pfa_pd_via_cfar_rrect(seg_bin_image, gt_rotate_rects_dict, seg_blob_list=[]):
    '''
    This function is used for the simulated Groundtruth.
    Prob. of false alarm and Prob. of detection via cfar for one frame.
    False Alarm Prob. is counted by pixels.
    Detection Prob. is computed by measuring the shrinked GT box (shrink 2 times).
    Shrinking aims to omit the large margins for the Gaussian Extended Target.

    NOTE: rrect means that ground truth are rotated rectangles.
          CFAR output is pixels of 1, or clustered regions, or rects.
    :param seg_bin_image:        binary segmentation of a frame (output of cfar)
    :param gt_rotate_rects_dict: contain all target's ground truth rotated rectangles.
           gt_rotate_rects_dict{'target_name':[x,y,w,h,theta]}
    :param seg_blob_list: segmented blobs (output of the connective regions segmentation)
    :return pfa_cfar,     pd_cfar,    (all count in pixels by  polygons)
            pfa_seg,      pd_seg (all count in pixels by  rectangles.)
    '''
    gt_mask            = np.zeros(seg_bin_image.shape, 'uint8')
    seg_mask           = np.zeros(seg_bin_image.shape, 'uint8')
    gt_seg_rect_pixels = 0  # ground truth positive pixels in segmented rectangles (shrink 2 times in size).
    tp_seg_rect_pixels = 0  # true positive of the segmented blob via cfar and pixels-cluster method in cv2.
    #fig, ax = plt.subplots()
    for key in gt_rotate_rects_dict:  # target name
        # target_region = gt_regions_dict[key]
        # format region vertex array  for cv2.fill
        rotated_rect       = gt_rotate_rects_dict[key]
        rx,ry,rw,rh,theta  = rotated_rect[:5]
        ## BEGIN - draw rotated rectangle
        #rect = cv2.minAreaRect(c)
        rotect_cv2 = ((rx + rw / 2, ry + rh / 2), (rw, rh), theta * 180 / np.pi)
        box_vertex = cv2.boxPoints(rotect_cv2)
        box_vertex = np.array(np.int0(box_vertex), 'int32')
        #Fill the gt polygon regions
        gt_mask    = cv2.fillConvexPoly(gt_mask, box_vertex, [1], lineType=None)
        ## END - draw rotated rectangle

        gt_box  = cv2.boundingRect(np.int0(box_vertex))
        # gt_positive += gt_box[2] * gt_box[3]
        gt_seg_rect_pixels += (gt_box[2] * gt_box[3])

        # x, y, w, h = gt_box[:4]
        # cx,cy = [int(x + w/2), int(y+h/2)]
        # #Shrink the gt box will increase the detection rate for clustering Gaussian Extended target.
        # gt_shrink_box = [int(cx-w/4), int(cy-h/4), int(w/2), int(h/2)]
        # gt_seg_rect_pixels += (gt_shrink_box[2]*gt_shrink_box[3])

        #from matplotlib.patches import Rectangle
        # ?gt_rr   = Rectangle(xy=[rx , ry], width=rw, height=rh, angle=theta*180/np.pi, color='white', fill=None)
        # gt_rect = Rectangle(xy=[x, y], width=w, height=h, angle=0, color='green', fill=None)
        # gt_shrk = Rectangle(xy=[int(cx-w/4), int(cy-h/4)], width=w/2, height=h/2, angle=0, color='red', fill=None)
        # ax.add_patch(gt_rr)
        # ax.add_patch(gt_rect)
        # ax.add_patch(gt_shrk)
        iou_max = 0
        tp_blob = []
        for seg_blob in seg_blob_list:
            #iou = uti.intersection_rect(seg_blob, gt_shrink_box)
            iou = uti.intersection_rect(seg_blob, gt_box)
            if iou > iou_max:
                iou_max = iou
                tp_blob = seg_blob
            # if iou >0:
            #     print('iou %.3f'%iou, tp_blob, gt_box, uti.intersection_area(tp_blob, gt_box))
        if iou_max > 0.15:  # iou is bigger enough.
            #tp_seg_rect_pixels += uti.intersection_area(tp_blob,gt_shrink_box)  # adding the true positive pixels of intersected rect.
            ##if gt_box is overlapped with a blob and iou>0.15, a extended target detection confirmed.
            tp_seg_rect_pixels  += gt_box[2] * gt_box[3]

    clustered_positive_pixels = 0 # clustered poisitive pixels in rectangle blobs.
    for seg_blob in seg_blob_list:
        bx, by, bw, bh = seg_blob[:4]
        seg_mask[by:by + bh, bx:bx + bw] = 1
    clustered_positive_pixels = np.sum(seg_mask)

    FP = np.sum(seg_bin_image) - np.sum(gt_mask * seg_bin_image)  # number of false positive pixels
    pfa_cfar = FP / (seg_bin_image.size - np.sum(gt_mask))
    pd_cfar  = np.sum(gt_mask * seg_bin_image) / np.sum(gt_mask)


    pfa_seg  = (clustered_positive_pixels - tp_seg_rect_pixels) / (seg_bin_image.size - np.sum(gt_mask))
    pd_seg   = min(tp_seg_rect_pixels, gt_seg_rect_pixels) / gt_seg_rect_pixels  # make sure less than 1.

    # ax.imshow(gt_mask)
    # plt.pause(0.1)
    # plt.show()
    if len(seg_blob_list) == 0:
        return pfa_cfar, pd_cfar
    else:
        return pfa_cfar, pd_cfar, pfa_seg, pd_seg

def get_pfa_pd_via_cfar(seg_bin_image, gt_regions_dict, seg_blob_list=[]):
    '''
    Prob. of false alarm and Prob. of detection via cfar for one frame
    p_fa = #(噪声>门限)/#(噪声)
    p_fa = #(seg_image - seg_image&gt)/#(wxh-gt)      FP/GN [False_Positive/Groundtruth_Negative]
    pd   = #(seg_image&gt) / #(gt)                    TP/GP [True_Positive/Groundtruth_Positive]
    pd   = #(跟踪矩形&gt)/gt

    NOTE: ground truth is polygon region or bounding box.
          CFAR output is pixels of 1, or clustered regions, or rects.
    :param seg_bin_image: binary segmentation of a frame
    :param gt_regions_dict:  contain all target's ground truth regions.
           gt_rects_dict{'target_name':[[x1,y1], [x2,y2],...]}
    :return:pfa_cfar,     pd_cfar,    (all count in pixels by  polygons)
            pfa_seg_blob, pd_seg_blob (all count in pixels by  rectangles.)
    '''
    gt_mask = np.zeros_like(seg_bin_image)
    seg_mask = np.zeros_like(seg_bin_image)
    gt_positive = 0  # ground truth positive pixels
    gt_region_positive = 0
    TP_blob_pixels = 0  # true positive of the segmented blob via cfar and pixels-cluster method in cv2.
    for key in gt_regions_dict:  # target name
        # target_region = gt_regions_dict[key]
        # format region vertex array  for cv2.fill
        target_region = np.array([np.int0(gt_regions_dict[key])])
        gt_mask = cv2.fillConvexPoly(gt_mask, target_region, [1], lineType=None)
        gt_box  = cv2.boundingRect(np.int0(gt_regions_dict[key]))
        #gt_positive += gt_box[2] * gt_box[3]

        x, y, w, h = gt_box[:4]
        target_roi = seg_bin_image[y:y + h, x:x + w]
        gt_positive = gt_positive + np.sum(target_roi)
        # gt_mask[y:y+h, x:x+w] = 1 # set the gt mask for all the targets.

        iou_max = 0
        tp_blob = []
        for seg_blob in seg_blob_list:
            iou = uti.intersection_rect(seg_blob, gt_box)
            if iou> iou_max:
                iou_max = iou
                tp_blob = seg_blob
            # if iou >0:
            #     print('iou %.3f'%iou, tp_blob, gt_box, uti.intersection_area(tp_blob, gt_box))
        if iou_max > 0.15: # iou is bigger enough.
            TP_blob_pixels += uti.intersection_area(tp_blob, gt_box) # adding the true positive pixels of intersected rect.
    #print(TP_blob_pixels, gt_positive, TP_blob_pixels/TP_rect)


    clustered_positive_pixels = 0
    for seg_blob in seg_blob_list:
        bx,by,bw,bh = seg_blob[:4]
        seg_mask[by:by+bh, bx:bx+bw] = 1
    clustered_positive_pixels = np.sum(seg_mask)

    FP = np.sum(seg_bin_image) - np.sum(gt_mask * seg_bin_image)  # number of false positive pixels
    pfa_cfar = FP / (seg_bin_image.size - np.sum(gt_mask))
    pd_cfar = np.sum(gt_mask * seg_bin_image) / np.sum(gt_mask)

    pfa_seg_blob = (clustered_positive_pixels - TP_blob_pixels)/(seg_bin_image.size - np.sum(gt_mask))
    pd_seg_blob  = min(TP_blob_pixels, gt_positive) / gt_positive #make sure less than 1.

    if len(seg_blob_list) == 0:
        return pfa_cfar, pd_cfar
    else:
        return pfa_cfar, pd_cfar, pfa_seg_blob, pd_seg_blob

def get_pfa_pd_via_trajectory(trk_traj, gt_traj, precision_dict, frame_height, frame_width):
    '''
    Get prob. of false alarm and prob. of detection from tracker's trajectory.
    trk_traj{fid1:{tid1:[x,y,w,h], tid2:[x2,y2,w2,h2],...}, fid2:{tid1:[x,y,w,h], ....}, ....}
    gt_traj{fid1:{tname1:[x,y,w,h], tname2:[]...}, fid2:{tname1:[x,y,w,h]...},...}
    precision_dict[target_name][trk_name] = {'ave_epos':, 'ave_ew':, 'ave_eh':, ...}
    :param trk_traj:
    :param gt_traj:
    :return:
    '''
    gt_mask  = np.zeros((frame_height, frame_width))  # gt_mask  should set all the pixels  in gt_rects  as 1
    trk_mask = np.zeros((frame_height, frame_width))  # trk_mask should set all the pixels  in trk_rects as 1
    match_trk_mask = np.zeros((frame_height, frame_width))
    fid_list = np.int0(list(gt_traj.keys()))
    pfa_list = []
    pd_list = []

    matched_tids = []  # select the matched tids in list
    falo_tids    = []  # tid matched for 'falo' target. Tracking results is longer than ground_truth.
    for gt_tname in precision_dict:
         matched_tids.extend(list(precision_dict[gt_tname].keys()))

    for fid in gt_traj:
        gt_mask        = gt_mask  * 0  # clear gt  mask in each frame.
        trk_mask       = trk_mask * 0  # clear trk mask
        match_trk_mask = match_trk_mask * 0  # clear trk mask
        for tname in gt_traj[fid]: # Check all the gt rect
            rect = gt_traj[fid][tname][:4]
            x, y, w, h = np.int0(rect)
            gt_mask[y:y + h, x:x + w] = 1  # set the pixels in  gt rects as 1.
        #Loop all the matched tracker and accumulate the track_mask.
        if fid in trk_traj:
            for tid in trk_traj[fid]:
                trk_rect = trk_traj[fid][tid]
                tx, ty, tw, th = trk_rect
                trk_mask[ty:ty + th, tx:tx + tw] = 1
                if tid in matched_tids: # this tid's tracker is matched with gt_traj in precision_dict.
                        match_trk_mask[ty:ty + th, tx:tx + tw] = 1  # set the pixels in  track rects as 1.

        pfa = (np.sum(trk_mask) - np.sum(trk_mask * gt_mask)) / (frame_height * frame_width - np.sum(gt_mask))
        # note here the true_positive is from the intersection of match_trk_mask and gt_mask
        pd  = np.sum(match_trk_mask * gt_mask) / np.sum(gt_mask)
        pfa_list.append(pfa)
        pd_list.append(pd)

    return fid_list, pfa_list, pd_list

def convert_target_trajectory_to_frame_trajectory(targets_trajectory, bshrink=False):
    '''
    Convert target_trajectory{tname:{fid:[x,y,w,h],...}, tname2:{fid:[x,y,w,h]}} to fid_trajectory{fid:{tname:[x,y,w,h]}, fid2:{}}
    :param targets_trajectory:
    :param bshrink: shrink the rectangles, due to MCF init kcftracker has enlarged the segmented blob.
    :return:
    '''
    fid_traj = {}
    fid_keys = []
    for tname in targets_trajectory:
        fid_keys.extend(list(targets_trajectory[tname].keys()))
    fid_keys = list(set(fid_keys)) #combine the same fids by set.
    for fid in fid_keys:
        fid_traj[fid] = {} # Make sure that key fid is int in the returned fid_traj dict.
        for tname in targets_trajectory:
            if fid in targets_trajectory[tname]:
                rect = targets_trajectory[tname][fid]
                if bshrink:
                    bx,by,bw,bh = rect[:4]
                    bcx = bx + bw/2
                    bcy = by + bh/2
                    bw  = 0.75*bw
                    bh  = 0.75*bh
                    fid_traj[fid][tname] = np.int0([bcx-bw/2, bcy-bh/2, bw, bh])
                else:
                    fid_traj[fid][tname] = rect
    return fid_traj

def reform_multiple_extended_targets_gt(met_dict):
    '''
    Input the multiple_extended_targets format: {['fid']:{['tid']:[cx, cy, w, h, theta]}}
    Transform to targets' trajectory dict. Each elements is a trajectory dict {'fid0':[rect], ..., 'fid50':[rect]} for a target.

    :param met_dict:
    :return:
    '''
    targets_traj_dict = {}
    for framekey in met_dict:
        met = met_dict[framekey]
        for targetkey in met:
            x, y, w, h = (met[targetkey])[:4]
            rect = [x, y, w, h ]
            if targetkey not in targets_traj_dict:
                targets_traj_dict[targetkey] = {}
            targets_traj_dict[targetkey][framekey] = rect #make sure that frame key is int in the returned targets_traj_dict
    return targets_traj_dict

def  move_figure(fig, x, y):
    """Move figure's upper left corner to pixel (x, y)
       Refer to: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """

    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig.canvas.manager.window.move(x, y)
