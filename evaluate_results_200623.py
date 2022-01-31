'''
Evalute the tracking results, compared to the gt rects frame by frame.
Two metric, position_precision and IoU rate are used.

2020-12-15 Add the multiple target gt_dict and matching functions.
"reform_multiple_extended_targets_gt()"
Add the Time on target(TOT), Track fragementations (TF)
'''
import pickle
import glob
import cv2
import numpy as np
from matplotlib.patches import Rectangle
import utilities_200611         as uti            # personal tools
import matplotlib.pyplot        as plt

def mark_trajectory(canvas, trajectory, color=(255,255,255)):
    '''
    mark the trajactory in color
    :param trajectory {'frame_no':rect_x,y,w,h}
    :return:canvas
    '''
    frame_h, frame_w = canvas.shape[:2]
    for frame_no in trajectory:
        rect = trajectory[frame_no]
        cx, cy = np.int0([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2])
        cx = min(cx, frame_w - 1)
        cy = min(cy, frame_h - 1)
        canvas[cy, cx, :] = color
    return canvas

def evalute_taes20sim_tracker():
    file_prefix = '/Users/yizhou/Radar_Datasets/TAES20_Simulate/gp_corrupted_9_0623/'
    gt_file = '/Users/yizhou/Radar_Datasets/TAES20_Simulate/gp_corrupted_9_0623/taes20_gt_9targets.pickle'
    vdf_file= '/Users/yizhou/Radar_Datasets/TAES20_Simulate/taes20_vdt_9targets.pickle'

    with open(gt_file, 'rb') as f:
        gt_dict = pickle.load(f)
    with open(vdf_file, 'rb') as f:
        vdt_dict = pickle.load(f)

    file_names = glob.glob(file_prefix + '*.png')
    file_names.sort()
    file_len = len(file_names)
    # view the gt and vdt tracking results visually
    # for i in range(0, file_len):
    #     fname_split = file_names[i].split('/')
    #     frame_no = int(fname_split[-1].split('.')[0])
    #     print('frame no %d' % frame_no)
    #     frame = cv2.imread(file_names[i], 0)  # gray_scale image
    #     canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #     frame_no_key = '%02d' % frame_no
    #     for key in gt_dict:
    #         uti.draw_rect(canvas, gt_dict[key][frame_no_key], color=(255, 255, 255))
    #     for vdtkey in vdt_dict:
    #         if frame_no_key in vdt_dict[vdtkey]:
    #             uti.draw_rect(canvas, vdt_dict[vdtkey][frame_no_key], color=(0, 255, 255))
    #     canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    #     plt.imshow(canvas)
    #     plt.pause(0.1)
    #     plt.draw()
    #    # plt.waitforbuttonpress()


    frame = cv2.imread(file_names[file_len-1], 0)  # last frame in gray_scale image
    frame_h, frame_w = frame.shape[:2]
    canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #view the trajectory
    for t in range(0,100):
        frame_no_key = '%02d' % t
        #mark the groudtruth
        for key in gt_dict:
            gt_rect = gt_dict[key][frame_no_key]
            cx, cy  = np.int0([gt_rect[0]+gt_rect[2]/2, gt_rect[1]+gt_rect[3]/2])
            cx = min(cx, frame_w-1)
            cy = min(cy, frame_h-1)
            canvas[cy, cx, :]=(255, 255, 255)   # ground truth in white
        for vdtkey in vdt_dict:
            if frame_no_key in vdt_dict[vdtkey]:
                vdt_rect = vdt_dict[vdtkey][frame_no_key]
                vcx, vcy = np.int0([vdt_rect[0] + vdt_rect[2] / 2, vdt_rect[1] + vdt_rect[3] / 2])
                vcx = min(vcx, frame_w-1)
                vcy = min(vcy, frame_h-1)
                canvas[vcy, vcx, :] = (0, 255, 255)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    plt.imshow(canvas)

    match_dict = match_trajectory(gt_dict, vdt_dict)
    #view the matched trajectory
    frame = cv2.imread(file_names[file_len - 1], 0)  # last frame in gray_scale image
    canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for target_name in match_dict:
         mark_trajectory(canvas, gt_dict[target_name], color=(255,255,255)) #ground truth in white
         for trk_name in match_dict[target_name]:
             #if len(vdt_dict[trk_name])>len(gt_dict[target_name])/2: #mark the long-term tracker
                mark_trajectory(canvas, vdt_dict[trk_name], color=(0,255,255))
                print('(Distance, IoU) between gt_%s and tid_%s is (%.2f, %.2f)'
                      %(target_name, trk_name,
                        match_dict[target_name][trk_name][0], match_dict[target_name][trk_name][1]))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(canvas)
    plt.show()

def dist_of_two_trajectory(gt_trajectory, trk_trajectory):
    '''
    compute the distance of two trajectory {'frame_no':[rect_x, y, w, h]}
    :param gt_trajectory:
    :param trk_trajectory:
    :return: total dist
    '''
    trk_rect_list = []
    gt_rect_list  = []
    for key in trk_trajectory:
        if key in gt_trajectory:
            trk_rect_list.append(trk_trajectory[key])
            gt_rect_list.append(gt_trajectory[key])
    if len(trk_rect_list)==0: # gt_trajectory is not available in the tracker's frame.
        ave_dist = 100000 # asign a big number for distance between gt_traj and trk_traj
        return ave_dist
    assert(len(trk_rect_list)>0 and len(gt_rect_list)>0)
    trk_rect_arr = np.array(trk_rect_list)
    gt_rect_arr  = np.array(gt_rect_list)
    dcx = (trk_rect_arr[:,0]+trk_rect_arr[:,2]/2)-(gt_rect_arr[:,0]+gt_rect_arr[:,2]/2)
    dcy = (trk_rect_arr[:,1]+trk_rect_arr[:,3]/2)-(gt_rect_arr[:,1]+gt_rect_arr[:,3]/2)
    dist = np.sqrt(dcx**2 + dcy**2)
    dist_sum = np.sum(dist)
    ave_dist = dist_sum/len(trk_rect_list)
    return ave_dist

def iou_of_two_trajectory(gt_trajectory, trk_trajectory):
    '''
        compute the distance of two trajectory {'frame_no':[rect_x, y, w, h]}
        :param gt_trajectory:
        :param trk_trajectory:
        :return: total dist
        '''
    iou_sum = 0
    nframes = 0
    ious    = []
    for key in trk_trajectory:
        if key in gt_trajectory:
            iou = uti.intersection_rect(trk_trajectory[key], gt_trajectory[key])
            iou_sum += iou
            nframes += 1
        else:
            iou = 0
        ious.append(iou)
    ave_iou = iou_sum / (nframes + np.spacing(1))
    return ave_iou, ious

def match_trajectory(gt_dict, vdt_dict):
    '''
    Match the trajectory between the ground truth and the vdt trace.
    :param gt_dict:  contains all the gt trajectory
    :param vdt_dict: contains all the tracker's trajectory
    :return: match_dict {target_name_gt: trk_tid}
    '''
    match_dict = dict()
    for target_name in gt_dict:
        match_dict[target_name] = {}  #initial the match dict.
    for trk_name in vdt_dict:         #loop all the target
        trk_trajectory = vdt_dict[trk_name]
        for target_name in gt_dict:
            gt_trajectory = gt_dict[target_name]
            ave_dist      = dist_of_two_trajectory(gt_trajectory, trk_trajectory)
            ave_iou       = iou_of_two_trajectory(gt_trajectory, trk_trajectory)
            #print('Distance between gt_%s and tid_%s is %.2f'%(target_name, trk_name, dist_sum))
            if ave_dist<=8: #each frame get less than ave 8 pixels distance
                match_dict[target_name][trk_name]= (ave_dist, ave_iou)
        #print('/n')
    #print(match_dict)
    return match_dict

def get_cle_per_frame(match_dict, gt_dict, vdt_dict, target_state_dict):
    '''
    Add Center Location Error from matched trajectory to the target_state_dict
    :param match_dict: matched pairs:[gt_name-tracked_traj]
    :param gt_dict : ground-truth trajectory.
    :param vdt_dict:trajectory of the tracker, including the position of each frame.
    :return:
    '''
    match_target_state_dict = {}
    for target_name in match_dict:
        gt_trajectory = gt_dict[target_name]
        match_target_state_dict[target_name] = {}
        for tid in match_dict[target_name]:
            trk_trajectory = vdt_dict[tid]
            trk_rect_list  = []
            gt_rect_list   = []
            fids           = []
            for key in trk_trajectory:
                if key in gt_trajectory:
                    fids.append(key)
                    trk_rect_list.append(trk_trajectory[key])
                    gt_rect_list.append(gt_trajectory[key])

            if (len(trk_rect_list) > 0) and (len(gt_rect_list) > 0):
                trk_rect_arr = np.array(trk_rect_list)
                gt_rect_arr  = np.array(gt_rect_list)
                dcx = (trk_rect_arr[:, 0] + trk_rect_arr[:, 2] / 2) - (gt_rect_arr[:, 0] + gt_rect_arr[:, 2] / 2)
                dcy = (trk_rect_arr[:, 1] + trk_rect_arr[:, 3] / 2) - (gt_rect_arr[:, 1] + gt_rect_arr[:, 3] / 2)
                dist = np.sqrt(dcx ** 2 + dcy ** 2)
                for n,fid in enumerate(fids):
                    # assign the center location error from the dist vector
                    target_state_dict[tid][fid]['cle'] = dist[n]
                match_target_state_dict[target_name][tid] = target_state_dict[tid].copy()
    return match_target_state_dict

def draw_target_state(match_target_state_dict, ax=None):
    '''
    draw the matched_tracker's psr,peak,psnr,cle in one figure.
    :param match_target_state_dict:
    :return:
    '''
    for tname in match_target_state_dict:
        fig,ax = plt.subplots()
        for tid in match_target_state_dict[tname]:
            fids = list(match_target_state_dict[tname][tid].keys())
            cles  = []
            psnrs = []
            star_peaks = []
            star_lifes = []
            star_psrs  = []
            ave_psrs   = []
            ave_lifes  = []
            int_lambdas = [] #integrated lambdas.
            for fid in match_target_state_dict[tname][tid]:
                cle = match_target_state_dict[tname][tid][fid]['cle']
                cles.append(cle)
                psnr = match_target_state_dict[tname][tid][fid]['psnr']
                psnrs.append(psnr)
                star_peak = match_target_state_dict[tname][tid][fid]['star_peak']
                star_peaks.append(star_peak)
                star_life = match_target_state_dict[tname][tid][fid]['star_life']
                star_lifes.append(star_life)
                star_psr = match_target_state_dict[tname][tid][fid]['star_psr']
                star_psrs.append(star_psr)
                ave_psr = match_target_state_dict[tname][tid][fid]['ave_psr']
                ave_psrs.append(ave_psr)
                ave_life = match_target_state_dict[tname][tid][fid]['ave_life']
                ave_lifes.append(ave_life)
                int_lambda = match_target_state_dict[tname][tid][fid]['int_lambda']
                int_lambdas.append(int_lambda)
        ax.plot(fids[:-2], cles[:-2], 'g', label='CLE(pixels)', lw=2, alpha=0.5)
        ax.plot(fids[:-2], psnrs[:-2], 'b--', label='PSNR(dB)', lw=1)
        ax.plot(fids[:-2], star_psrs[:-2], 'r-.',  label ='PSR $%s$'% r'(s_{i^*})')
        ax.plot(fids[:-2], int_lambdas[:-2], 'y+', label='$%s$'% r'\sum\lambda_k', lw=2)
        ax.set_xlabel('Frame #')
        plt.grid()
        plt.legend()
        fig.savefig('/Users/yizhou/code/taes2021/results/' + 'Titan_psr_cle.pdf', format='pdf',
                         dpi=300,
                         bbox_inches='tight')
        plt.show()
        print('')

def precision_of_extended_target(gt_trajectory, trk_trajectory):
    '''
    compute the distance of two trajectory {'frame_no':[rect_x, y, w, h]}
    return the mean square error on center position and width and height.
    :param gt_trajectory:
    :param trk_trajectory:
    :return: total dist
    '''
    trk_rect_list = []
    gt_rect_list  = []
    start_fid     = -1
    end_fid       = -1
    fids          = []
    for key in trk_trajectory:
        if key in gt_trajectory:
            trk_rect_list.append(trk_trajectory[key])
            gt_rect_list.append(gt_trajectory[key])
            fids.append(int(key))

    if len(gt_rect_list)==0: # no gt_traj available for trk_traj, return big error
        ave_position_error = 100000
        ave_ew             = 100000
        ave_eh             = 100000
        epos_dist          = []
        return ave_position_error, ave_ew, ave_eh, epos_dist, start_fid, end_fid

    start_fid = min(fids)
    end_fid   = max(fids)

    trk_rect_arr = np.array(trk_rect_list)
    gt_rect_arr = np.array(gt_rect_list)
    dcx = (trk_rect_arr[:, 0] + trk_rect_arr[:, 2] / 2) - (gt_rect_arr[:, 0] + gt_rect_arr[:, 2] / 2)
    dcy = (trk_rect_arr[:, 1] + trk_rect_arr[:, 3] / 2) - (gt_rect_arr[:, 1] + gt_rect_arr[:, 3] / 2)
    epos_dist = np.sqrt(dcx ** 2 + dcy ** 2)
    epose_sum = np.sum(epos_dist)
    ave_position_error = epose_sum / len(trk_rect_list)

    dw = np.abs(trk_rect_arr[:, 2] - gt_rect_arr[:, 2])
    dh = np.abs(trk_rect_arr[:, 3] - gt_rect_arr[:, 3])
    ave_ew = np.sum(dw)/len(trk_rect_list)
    ave_eh = np.sum(dh)/len(trk_rect_list)
    return ave_position_error, ave_ew, ave_eh, epos_dist, start_fid, end_fid

def measure_trajectory_precesion(gt_dict, vdt_dict):
    '''
    Based on the match_dict{'gt_id1':{'tid1':[e_position, iou], 'tid2':[e_position, iou]}} in function match_trajectory()
    add more measurements such as: time on gt_target, frack fragementation, e_width, e_height.
    to get a new precision matrix, each gt_target may have multiple related trackers.
    :param match_dict:
    :param tracker_list:
    :return:
    '''
    precision_dict = dict()
    false_alarm_acc    = 0  #accumulated false alarm
    matched_tids       = [] #used to findout the unmatched trackers for false alarm.
    for target_name in gt_dict:
        precision_dict[target_name] = {}  #initial the match dict.
    for trk_name in vdt_dict:         #loop all the target
        trk_trajectory = vdt_dict[trk_name]
        for target_name in gt_dict:
            gt_trajectory = gt_dict[target_name]
            ave_position_error, ave_ew, ave_eh, epos_dist,start_fid, end_fid  = precision_of_extended_target(gt_trajectory, trk_trajectory)
            ave_iou, ious  = iou_of_two_trajectory(gt_trajectory, trk_trajectory)
            #print('Distance between gt_%s and tid_%s is %.2f'%(target_name, trk_name, dist_sum))
            if ave_position_error<=(8): #each frame get less than ave 8 pixels distance
                #precision_dict[target_name][trk_name]= (ave_position_error, ave_ew, ave_eh, ave_iou)

                # start_fid = 100000000
                # end_fid   = -1
                # #find the maximum and minimum fid.
                # for fid_key in trk_trajectory:
                #     fid  = int(fid_key)
                #     if fid < start_fid:
                #         start_fid = fid
                #     if fid > end_fid:
                #         end_fid   = fid
                #rate_of_hit = (end_fid-start_fid+1)/len(gt_trajectory)
                precision_dict[target_name][trk_name] = {'ave_epos': ave_position_error,
                                                 'ave_ew':ave_ew,
                                                 'ave_eh':ave_eh,
                                                 'ave_iou': ave_iou,
                                                 'start_fid': start_fid,
                                                 'end_fid': end_fid,
                                                 'epos_dist':epos_dist,
                                                 'ious':ious}
                matched_tids.append(trk_name)
    tracked_tids = vdt_dict.keys()
    for tid in tracked_tids:
        if tid not in matched_tids : # False alarm produced Tracker.
            trk_trajectory = vdt_dict[tid]
            for fid in trk_trajectory:
                x,y,w,h = trk_trajectory[fid]
                false_alarm_acc += w*h
    ## compute the roh
    ## rate of hit,  the ratio between correct tracked tail points and all the gt tail points.

    overall_roh_dict = {}
    for target_name in precision_dict:
        fids = []
        for trk_name  in precision_dict[target_name]:
            trk_trajectory =  vdt_dict[trk_name]
            fids += trk_trajectory.keys()  # accumulate all the fids in each matched trajectory
            #make sure that the roh is not bigger than 1.
            precision_dict[target_name][trk_name]['roh'] = min(len(trk_trajectory), len(gt_trajectory)) / len(gt_trajectory)
        non_repeat_fids = list(set(fids))
        gt_trajectory = gt_dict[target_name]

        roh = min(len(non_repeat_fids),len(gt_trajectory)) / len(gt_trajectory)
        overall_roh_dict[target_name] = roh

    return precision_dict, false_alarm_acc, overall_roh_dict

def get_track__precesion(precision_dict, overall_roh_dict):
    '''
    Get precision of each target.
    :return:
    '''

    gt_names = precision_dict.keys()
    res_trk_precision_dict = {}
    ave_epos = 0
    ave_iou  = 0
    ave_roh  = 0
    ave_ntf  = 0
    m = len(gt_names) #+ np.spacing(1)
    precision_matrix = np.zeros((4, m+1)) # rows are the metric: epos, iou, roh, ntf
    cols = 0
    for name in gt_names:
        #record error_pos, iou and roh for each matched gt_target
        res_trk_precision_dict[name] = {}
        epos = 0
        iou = 0
        roh = 0
        ntf = 0  # number of tracker fragmentation.
        #add column for each gt target
        for tid in precision_dict[name]:
            epos += precision_dict[name][tid]['ave_epos']
            iou  += precision_dict[name][tid]['ave_iou']
            #roh  += precision_dict[name][tid]['roh']
        roh = overall_roh_dict[name]
        ntf = len(precision_dict[name]) #gt traj contains how many independent tracker.
        n = max(1,len(precision_dict[name]))#+ np.spacing(1)
        ave_epos += epos/n
        ave_iou  += iou/n

        res_trk_precision_dict[name]['ave_epos'] = ave_epos
        res_trk_precision_dict[name]['ave_iou']  = ave_iou
        res_trk_precision_dict[name]['ntf']      = ntf
        res_trk_precision_dict[name]['roh']      = roh
    return res_trk_precision_dict

def print_metrics(precision_dict, false_alarm_acc, image_width, image_height, nframes, overall_roh_dict):
    '''
    Print the metrics results in a table. Same as TableIII in Paper IJOEVivone2016
    :return:
    '''
    str_table= {}
    gt_names = precision_dict.keys()
    str_table_title = '%12s'%'' # emtpy unit in a table
    for name in gt_names:
        str_table_title += '%10s'%name
    str_table_title += '%12s'%'Ave-Res.'
    #str_table.append(str_table_title)
    #print(str_table_title) #print table title

    str_table_row  = ''
    epos_list = []
    iou_list  = []
    roh_list  = []
    ntf_list  = []
    ave_res_list = []
    ave_epos = 0
    ave_iou  = 0
    ave_roh  = 0
    ave_ntf  = 0
    m = len(gt_names) #+ np.spacing(1)
    precision_matrix = np.zeros((4, m+1)) # rows are the metric: epos, iou, roh, ntf
    cols = 0
    for name in gt_names:
        #record error_pos, iou and roh for each matched gt_target
        epos = 0
        iou = 0
        roh = 0
        ntf = 0  # number of tracker fragmentation.
        #add column for each gt target
        str_table[name] = {}
        for tid in precision_dict[name]:
            epos += precision_dict[name][tid]['ave_epos']
            iou  += precision_dict[name][tid]['ave_iou']
            #roh  += precision_dict[name][tid]['roh']
        roh = overall_roh_dict[name]
        ntf = len(precision_dict[name]) #gt traj contains how many independent tracker.
        n = max(1,len(precision_dict[name]))#+ np.spacing(1)

        #
        #str_table[name]['ave_epos'] = epos/n
        #the upbounding mismatch position errors are 50 pixels.
        #increase the epos to 50, in those no-matched-tracker frames.
        str_table[name]['ave_epos'] = roh*(epos / n) + (1-roh)*50
        #str_table[name]['ave_iou' ] = iou/n
        str_table[name]['ave_iou']  = roh*iou / n    + (1-roh)*0
        str_table[name]['roh']      = roh
        str_table[name]['ntf']      = ntf
        # ave_epos += epos/n
        # ave_iou  += iou/n
        ave_epos += roh*(epos/n) + (1-roh)*50
        ave_iou  += roh*iou/n    + (1-roh)*0
        ave_roh  += roh
        ave_ntf  += ntf
        precision_matrix[:, cols] = [epos/n, iou/n, roh, ntf]
        cols += 1
    #add new column for ave_res
    str_table['ave_res']= {}
    str_table['ave_res']['ave_epos'] = ave_epos/m
    str_table['ave_res']['ave_iou']  = ave_iou /m
    str_table['ave_res']['roh']      = ave_roh /m
    str_table['ave_res']['ntf']      = ave_ntf /m
    # las column is the average.
    precision_matrix[:,-1] = [ave_epos/m, ave_iou/m, ave_roh/m, ave_ntf/m]

    row_heads    = ['ave_epos', 'ave_iou', 'roh', 'ntf']
    nrows        = len(row_heads)  #
    ntargets     = len(gt_names)   # Number of GT Targets.
    print(str_table_title)
    for i in range(nrows):
        str_table_row = '%10s'%row_heads[i]
        metric_name   = row_heads[i]
        for name in str_table:
            str_table_row += '%10.2f' % str_table[name][metric_name]
        print(str_table_row)

    false_alarm_rate = false_alarm_acc / (image_height * image_width * nframes)

    print('False alarm rate %.2E'%false_alarm_rate)
    #print(precision_matrix)
    return precision_matrix, false_alarm_rate, str_table




def draw_trajectory(ax, trajectory_dict, name, color_tuple):
    '''
    Draw trajectory_dict on ax in color_tuple, mark the name in 1st points.
    :param ax:
    :param trajectory_dict:
    :param color_tuple = (R, G, B, Alpha) in float value (0~1):
    :return:
    '''
    index = 0
    for fid in trajectory_dict:
        x, y, w, h = trajectory_dict[fid]
        if index == 0:  # first element
            ax.text(x, y, name, color=(1,1,1,1), fontsize=6)
        index += 1
        rect_patch = Rectangle(xy=[x+w/2, y+h/2], width=2, height=2, angle=0, color=color_tuple, fill=None)
        ax.add_patch(rect_patch)

def draw_track_traj(img_w, img_h, gt_dict, vdt_dict, precision_dict):
    '''
    draw tracker's trajectory in different color to monitor the trajectory fragmentation.
    :return:
    '''
    print('Gt target numbers is %d' %(len(gt_dict)))
    canvas = np.zeros((img_h, img_w, 3))

    fig, ax = plt.subplots()
    ax.imshow(canvas)


    # Draw white circle for the ground truth Target.
    for target_name in gt_dict:
        gt_trajectory = gt_dict[target_name]
        draw_trajectory(ax, gt_trajectory, target_name, color_tuple=(1,1,1,1)) # color_tuple is RGBA tuple.

    # Only draw the matched trajectory
    # for gt_name in precision_dict:
    #     matched_trackers_dict = precision_dict[gt_name]
    #     matched_trackers_num  = len(matched_trackers_dict)
    #     # 0 blue, 1 green, 2 red,
    #     color_options = [(1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1), (0,1,1,1), (1,0,1,1)]
    #     index = 0
    #     #draw matched_trackers in 6 cololrs.
    #     for tid in matched_trackers_dict:
    #         trk_trajectory = vdt_dict[tid]
    #         draw_trajectory(ax, trk_trajectory, tid, color_tuple= color_options[index % 6])
    #         index +=1
    # Draw all the trajectory.

    # Draw random color for the all the tracker trajectories.
    for trk_tid in vdt_dict:
        trk_trajectory = vdt_dict[trk_tid]
        draw_trajectory(ax, trk_trajectory, trk_tid, color_tuple=np.random.random(3).tolist())
        #plt.pause(0.01)
        #plt.waitforbuttonpress()

    #plt.show()

def draw_rmse(precision_dict):
    '''
    Draw root mean square error VS. fids.
    :return:
    '''
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('rmse')
    for tname in precision_dict:
        for tid in precision_dict[tname]:
            rmse      = precision_dict[tname][tid]['epos_dist']
            start_fid = precision_dict[tname][tid]['start_fid']
            end_fid   = precision_dict[tname][tid]['end_fid'  ]
            fids      = np.arange(start_fid, end_fid+1, 1)
            ltext     = '%s-%d(%.2f)'%(tname, tid, np.mean(rmse))
            ax.plot(fids, rmse, label=ltext)
    ax.legend()
    plt.pause(0.01)

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

def draw_precision_curve(res_dict):
    '''
    Plot the figure based on res_dict{'tracker_name':{'epos_rms':epos_rms, 'ious':ious,
                                               'iou_precision': iou_precision,'frame_nos':fids}}
    tracker_name are :['CF-TBD', 'PT-TBD', 'PS-TBD', 'ET-JPDA']
    :return:
    '''
    #prepare figure parameters
    line_style = ['-', '-', '--']
    marker     = ['o', '>', '1']
    alpha_vals = [0.4, 0.5, 1, 1, 1, 1, 1]
    colors      = ['r', 'g', 'b']
    params = {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 2,
        'legend.fontsize': 10,
        # 'figure.figsize': '12, 9'  # set figure size
    }
    plt.rcParams.update(params)  # set figure parameter

    label_font = 10
    legend_font= 10

    # figure for the center_location_error.
    cle_fig = plt.figure()
    cle_ax  = cle_fig.add_subplot(111)
    plt.xlabel('Frame', fontsize=label_font)
    plt.ylabel('RMSE(pixels)', fontsize=label_font)
    cle_ax.grid(True)

    # figure for the center_location_error.
    precle_fig = plt.figure()
    precle_ax  = precle_fig.add_subplot(111)
    #precle_ax  = precle_fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    plt.xlabel('RMSE threshold(pixels)', fontsize=label_font)
    plt.ylabel('Location Precision', fontsize=label_font)
    precle_ax.grid(True)


    iou_fig = plt.figure()
    iou_ax  = iou_fig.add_subplot(111)
    plt.xlabel('Frame', fontsize=label_font)
    plt.ylabel('IOU', fontsize=label_font)
    iou_ax.grid(True)

    # figure for the overlap precision of IoU.
    preiou_fig = plt.figure()
    preiou_ax = preiou_fig.add_subplot(111)
    plt.xlabel('IOU threshold')
    plt.ylabel('Overlap Precision')
    preiou_ax.grid(True)

    for i, tname in enumerate(res_dict): # tracker's name
        # draw cle_figure.
        epos =res_dict[tname]['epos_rms']
        fids =res_dict[tname]['frame_nos']
        if len(epos)>0:
            prune_epos = epos.copy()
            prune_epos[(prune_epos > 50)] = 50
            cle_ax.plot(fids, prune_epos, line_style[i], label='%08s[%2.2f]'%(tname, np.mean(epos)),
                        color=colors[i],linewidth=1.5, alpha=alpha_vals[i], marker = marker[i], markersize=6, markevery=i+1)
            cle_ax.set_ylim([0, 51])
            #cle_ax.set_title(tname, fontsize=label_font)
            cle_ax.legend(loc="upper left", fontsize=legend_font)

        pos_precisions = res_dict[tname]['pos_precision']
        if  len(pos_precisions)>0:
            precle_ax.plot(np.arange(0, 50, 1).tolist(), pos_precisions, line_style[i], color=colors[i],
                           label=tname + '[%2.2f]' % np.mean(pos_precisions), linewidth=1.5, alpha=alpha_vals[i], marker=marker[i],markersize=6,markevery=i+1)
            precle_ax.legend(loc="upper right", fontsize=legend_font)  # set legend location
            #precle_ax.set_title(tname, fontsize=label_font)
            precle_ax.legend(loc="lower right", fontsize=legend_font)


        ious = res_dict[tname]['ious']
        if len(ious)>0:
            iou_ax.plot(fids, ious, line_style[i], label='%08s[%2.2f]'%(tname, np.mean(ious)),
                        color=colors[i],linewidth=1.5, alpha=alpha_vals[i], marker = marker[i],markersize=6,markevery=i+1)
            #iou_ax.set_ylim([0, 1.2])
            #iou_ax.set_title(tname, fontsize=label_font)
            iou_ax.legend(loc="upper right", fontsize=legend_font)

        iou_precisions = res_dict[tname]['iou_precision']
        if  len(iou_precisions)>0:
            preiou_ax.plot(np.arange(1, 0, -0.01).tolist(), iou_precisions, line_style[i], color=colors[i],
                           label=tname + '[%2.2f]' % np.mean(iou_precisions), linewidth=1.5, alpha=alpha_vals[i], marker=marker[i],markersize=6,markevery=i+1)
            preiou_ax.legend(loc="upper right", fontsize=legend_font)  # set legend location
            #preiou_ax.set_title(tname, fontsize=label_font)
        print('%s Average ErrorPositionRMS %.2f, Average IoU %.2f.'%(tname, np.mean(epos), np.mean(ious)))
    cle_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_cle.pdf', format='pdf',dpi = 300, bbox_inches='tight')
    cle_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_cle.png', format='png', dpi=300, bbox_inches='tight')
    iou_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_iou.pdf', format='pdf',dpi = 300, bbox_inches='tight')
    iou_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_iou.png', format='png', dpi=300, bbox_inches='tight')
    precle_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_precle.pdf', format='pdf',dpi = 300, bbox_inches='tight')
    precle_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_precle.png', format='png', dpi=300, bbox_inches='tight')
    preiou_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_preiou.pdf', format='pdf',dpi = 300, bbox_inches='tight')
    preiou_fig.savefig('/Users/yizhou/code/taes2021/results/'+'inesa'+'_preiou.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

def precision_cle_iou(gt_file_name, trk_file_name):
    '''
    compute the center location error and IoU based on the tracked_result file {fid:rect, ...}
    and gt_rect_dict file.
    :return:
    '''
    #1 position_RMS
    # # loading the Gt_rect first. '.\\results\\taes20_gt_titan_rect.pickle'
    # file_prefix = '/Users/yizhou/code/taes2021/results/'
    try:
        with open(gt_file_name, 'rb') as f:
            gt_traj = pickle.load(f)
        with open(trk_file_name, 'rb') as f:
            trk_traj = pickle.load(f)
    except Exception as e:
        gt_traj  = []
        trk_traj = []
        print(e)
    trk_rect_list = []
    gt_rect_list  = []
    fids          = []
    ious          = []
    iou_precision = []
    epos_dist     = []

    if len(gt_traj)==0 or len(trk_traj)==0: # return empty list.
        return epos_dist, ious, iou_precision, fids

    for key in trk_traj:
        if key in gt_traj: # Gets same key.
            trk_rect_list.append(trk_traj[key])
            gt_rect_list.append(gt_traj[key])
            iou = uti.intersection_rect(trk_traj[key], gt_traj[key])
            ious.append(iou)
            fids.append(key)

    trk_rect_arr = np.array(trk_rect_list)
    gt_rect_arr  = np.array(gt_rect_list)
    dcx = (trk_rect_arr[:, 0] + trk_rect_arr[:, 2] / 2) - (gt_rect_arr[:, 0] + gt_rect_arr[:, 2] / 2)
    dcy = (trk_rect_arr[:, 1] + trk_rect_arr[:, 3] / 2) - (gt_rect_arr[:, 1] + gt_rect_arr[:, 3] / 2)
    epos_dist = np.sqrt(dcx ** 2 + dcy ** 2) # position error (RMS)

    ious = np.array(ious)
    iou_threash_range = np.arange(1, 0, -0.01).tolist()
    iou_precision = np.zeros(len(iou_threash_range), np.float)
    for i, threash in enumerate(iou_threash_range):
        tracked_nums = np.sum(ious >= threash, dtype=float)
        iou_precision[i] = tracked_nums / len(ious)
    return epos_dist,ious, iou_precision, fids

def evaluate_inesa_tracking_results():
    '''
    Read the gt_traj and trackers's traj from all the pickle files.
    Computing the error of position frame by frame as epos,
    Computing the iou of frame by frame, making threshold to draw the precision_iou figure.
    all needs information are stored in res_dict{'track_name':{'epos_rms':epos_rms, 'ious':ious,
                                               'iou_precision': iou_precision,'frame_nos':fids}}
    :return res_dict
    '''
    res_dict = {}
    #Note in the paper of TAES2021, LELR-TBD has changed the name as 'WTSA-TBD'.
    tracker_name_list = ['MKCF-TBD','WTSA-TBD', 'MSAR-TBD']

    gt_file_name       =  '/Users/yizhou/code/taes2021/results/taes20_gt_titan_rect.pickle'
    with open(gt_file_name, 'rb') as f:
        gt_traj = pickle.load(f)

    #for titan
    tracker_precision_files = ['/Users/yizhou/code/taes2021/results/taes20_Titan_MKCF-TBD_precision_03_26.pickle',
                               '/Users/yizhou/code/taes2021/results/taes20_Titan_LELR-TBD_precision_03_26.pickle',
                               '/Users/yizhou/code/taes2021/results/taes20_Titan_MSAR-TBD_precision_03_26.pickle']
    #for trifalo
    # tracker_precision_files = ['/Users/yizhou/code/taes2021/results/taes20_TriFalo_MKCF-TBD_precision_03_29.pickle',
    #                            '/Users/yizhou/code/taes2021/results/taes20_TriFalo_LELR-TBD_precision_03_29.pickle',
    #                            '/Users/yizhou/code/taes2021/results/taes20_TriFalo_MSAR-TBD_precision_03_29.pickle']

    nframes = len(gt_traj.keys())
    fids    = range(1,nframes+1)

    for tracker_name, tracker_res_fname in zip(tracker_name_list, tracker_precision_files):
        with open(tracker_res_fname, 'rb') as f:
            precision_dict = pickle.load(f)
            res_dict[tracker_name] = {}
            for tname in precision_dict:
                if tname != 'Titan': # only take one target
                    continue
                res_dict[tracker_name][tname] = {}
                epos_dist = np.ones( (nframes, ))*50 # the default mismatch is 50 fixels
                ious      = np.zeros((nframes, ))

                for tid in precision_dict[tname]: # Loop all the track fragementation
                    start_fid = precision_dict[tname][tid]['start_fid'] #check the start fid of mkcf and msar
                    end_fid   = precision_dict[tname][tid]['end_fid']
                    epos_dist[start_fid-1:end_fid] = precision_dict[tname][tid]['epos_dist']
                    ious[start_fid-1:end_fid]      = precision_dict[tname][tid]['ious']
                ious[ious>1]=1 #avoid overlow
                #computing position precision, based on varying threshold.
                epos_dist = np.array(epos_dist)
                pos_threash_range = np.arange(0, 50, 1).tolist()
                pos_precision = np.zeros(len(pos_threash_range), np.float)
                for i, threash in enumerate(pos_threash_range):
                    tracked_nums = np.sum(epos_dist <= threash, dtype=float)
                    pos_precision[i] = tracked_nums / len(ious)

                #computing iou precision, based on varying threshold.
                ious = np.array(ious)
                iou_threash_range = np.arange(1, 0, -0.01).tolist()
                iou_precision = np.zeros(len(iou_threash_range), np.float)
                for i, threash in enumerate(iou_threash_range):
                    tracked_nums = np.sum(ious >= threash, dtype=float)
                    iou_precision[i] = tracked_nums / len(ious)


            res_dict[tracker_name]['epos_rms']      = epos_dist
            res_dict[tracker_name]['pos_precision'] = pos_precision
            res_dict[tracker_name]['ious']          = ious
            res_dict[tracker_name]['iou_precision'] = iou_precision
            res_dict[tracker_name]['frame_nos']     = fids

    draw_precision_curve(res_dict)



if __name__=='__main__':

    #evalute_taes20sim_tracker()
    evaluate_inesa_tracking_results()
    print('')
