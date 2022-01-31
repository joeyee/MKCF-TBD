'''
Compare three algorithms of tracking in the Raleigh distributed clutter: MCF, Grossi and LELR
with same input frames and same tracker framemork.
This file is based on
'MCF_TBD_Param_Test_Simulated_3targets_20210208.py'
'DP_TBD_Grossi_ETTsim_20201229.py'
'DP_TBD_LELR_ETTsim_20210304.py'

1st test the MCF's gt_tracker only(Initiate only the gt targets) to monitoring the status
and merit value changes across the frame.

'''

from scipy import ndimage
from scipy.stats import norm,rayleigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import utilities_200611         as uti            # personal tools
import cfar_segmentation_200527 as cfar_model     # cfar
from skimage.feature import local_binary_pattern
import cv2
from PIL import Image
import pickle
import glob

import numpy as np                  # matrix operation in Python
import glob                         # image files to list
import matplotlib.pyplot as plt     # observing plots
from matplotlib.lines import Line2D # line plot on image

import cv2                          # Fast operating images.
from sklearn.cluster import KMeans  # Simple clustering method test

from skimage.feature import local_binary_pattern
                                    # Local Binary Pattern function
from skimage import transform

from scipy.stats import itemfreq    # calculate a normalized histogram
from scipy import ndimage           # get filters

import pandas as pd                 # file records in/output
import pickle                       # save\load file in python data structure

# Python model and tools written by myself.
#import mosse_tracker_20200601     as mosse_model    # Single correlated tracker
import cfar_segmentation_200527   as cfar_model     # cfar
#import svm_learning_200610        as svm_model      # customized svm
#import vessel_detect_track_200614 as vdt_model      # jdat's ancestor
import evaluate_results_200623    as eval_model     # evaluate the tracking results.
import motion_simulation_20201030 as msim_model     # Motion simulation of multiple targets.
#import KCFtracker_Status_MotionVector as kcf_model
#import MCF_20200603              as mcf_model     # Fuse multiple correlated trackers for a target
import MCF_TBD_20201223           as mcf_model      # Fuse multiple correlated trackers for a target
import taes2021_utility_20210216  as sp_model       # support tools for TAES2021
import time
import csv
import matplotlib.patches         as mpatches
import matplotlib.lines           as mlines
import DP_TBD_Grossi_ETTsim_20201229 as gross_tbd_model
import DP_TBD_LELR_ETTsim_20210304   as lelr_tbd_model
#import DP_TBD_WTSA_ETTsim_20210514   as wtsa_tbd_model #using orientated gaussian coefficients
from datetime import datetime

#Set all the cfar Parameters for avoiding repeating in each tracker.
#parameters setting for cfar operation and cluster
cfar_seg_max_area       = 30000
cfar_seg_min_area       = 90
cfar_ref                = 16 * 2
cfar_guide              =  8 * 2
cfar_seg_kval           = 1.3#1.3

# field names
record_fields = ['SwerlingType', 'SNR', 'TimeCost', 'Pfa_cfar', 'Pfa_seg', 'Pfa_trk', 'Pfa_gain',
          'pd_cfar', 'pd_seg', 'pd_trk', 'pd_gain',
          'IntegratedFrames', 'Gamma', 'cfar_kval', 'ref_cells', 'guide_cells',
          'ave_epos', 'ave_iou', 'ave_roh', 'ave_ntf',
          'vic_epos', 'vic_iou', 'vic_roh', 'vic_ntf',
          'ame_epos', 'ame_iou', 'ame_roh', 'ame_ntf',
          'uri_epos', 'uri_iou', 'uri_roh', 'uri_ntf']

class MCF_TBD_SIM(): #MCF_TBD Tracker implemented for simulated situations.
    def __init__(self,img_w, img_h, start_fid, end_fid,
                      integrated_frames=9, integrated_merits_gamma=0.8,
                      ksigma=0.025, bVerbose=False):

        #simulated image size
        self.img_w                   = img_w
        self.img_h                   = img_h
        #parameters setting for MCF_TBD
        self.integrated_frames       = integrated_frames
        self.integrated_merits_gamma = integrated_merits_gamma
        self.ksigma                  = ksigma                    # Gaussian kernel sigma
        #show debug information or not, show trajectory figures or not
        self.bVerbose                = bVerbose
        if (self.bVerbose):
            self.fig = plt.figure(figsize=(7, 7))
            self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (800, 0))
            # New axis over the whole figure, no frame and a 1:1 aspect ratio
            self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
            self.ax.axis('off')


        #set the start and end frame_no.
        self.init_frame              = start_fid #start frame no.
        self.last_frame              = end_fid   # the last frame no.
        # snr   = 0         # snr-ksigma(20, 0.12) (10, 0.045) (3, 0.023) (0, 0.023)
        self.fid_list = []
        # name of csv file
        date = datetime.now()
        filename = "/Users/yizhou/code/taes2021/results/MCF_SIMULATE_RECORDS_%02d_%02d.csv"%(date.month, date.day)
        # writing to csv file
        self.csvfile   = open(filename, 'w')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(record_fields)

    def set_new_record_file(self, iter):
        '''
        each iteration gets one record file.
        :param iter:
        :return:
        '''
        self.csvfile.close()
        date = datetime.now()
        filename = "/Users/yizhou/code/taes2021/results/MCF_SIMULATE_RECORDS_%02d_%02d_%02d.csv"%(date.month, date.day, iter)
        # writing to csv file
        self.csvfile   = open(filename, 'w')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(record_fields)


    def set_gt_dict(self,gt_rotate_rect_trajectory_dict, gt_trajecotry_dict):
        '''
        Assign the ground-truth dict information.
        :param gt_rotate_rect_trajectory_dict:
        :param gt_trajecotry_dict:
        :return:
        '''
        self.gt_rotate_rect_trajectory_dict = gt_rotate_rect_trajectory_dict
        self.gt_trajecotry_dict             = gt_trajecotry_dict

    def clear_info_list(self):
        #clear the information list
        msim_model.local_snrs        = []  # clear the  local snrs records
        msim_model.global_snrs       = []  # clear the  msim_model's global snr records.

        self.fid_list                = []
        self.pfa_cfar_list           = []
        self.pfa_seg_list            = []
        self.pfa_trk_list            = []

        self.pd_cfar_list            = []
        self.pd_seg_list             = []
        self.pd_trk_list             = []

        self.mcf_tracker_list        = []
        self.time_counter_list       = []
        # save the qualified [mctracker.tracked_Frames > nums_thresh] target trajectory in the dict.
        # This dict's format is the same as the gt_dict in 'simulate_clutter_target_*.py'
        # {'target_name':{frame_no:[rect_x, y, w,h]}}
        self.target_trajectory_dict  = {}

    def draw_bounding_box(self, frame, frame_no, ax, blob_list, mcf_tracker_list, snr):
        '''
        Draw segmented blobs and tracked rectangles on the raw frame.
        :param frame:
        :param blob_list:
        :param tracker_list:
        :return:
        '''
        self.fig.canvas.set_window_title('frame%3d' % frame_no)
        ax.clear()
        ax.imshow(frame)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        #draw segmented blobs.
        ax.text(10, 20, 'SNR=%ddb'%snr, fontsize=18, fontweight='bold', color=(1, 1, 1, 1))
        for fid in range(frame_no):
            gt_target_dict = self.gt_trajecotry_dict[fid]
            for targrt in gt_target_dict:
                tx,ty,tw,th = gt_target_dict[targrt]
                circ = mpatches.Circle(xy=(tx+tw/2, ty+th/2), radius=1,color='white')
                ax.add_patch(circ)

            frame =1
        for blob in blob_list:
            tlx, tly, bw, bh = blob
            rect = mpatches.Rectangle(xy=[tlx, tly], width=bw, height=bh,
                             color=(0,1,1,0.8), lw=2, fill=None) #color r,g,b, blob in light blue.
            ax.add_artist(rect)
        #draw the rectangle of stored trackers
        for tracker in mcf_tracker_list:
            color_tuple = np.random.random(3).tolist()  # mark target using random color
            #Mark the tails of the target
            for id, target_rect in enumerate(tracker.tail_rect_list):
                [x, y, w, h] = target_rect[:4]
                if id == 1:  # First node write down the tid in the origin
                #     # tbox = [x - int(w / 2), y - int(h / 2), w, h]
                     ax.text(x+w/2, y+h/2, '%d' % tracker.tid, color=(1, 1, 1, 1), fontsize=6)
                tail_patch = mpatches.Rectangle(xy=[x + w/2 - 1, y + h/2 - 1], width=2, height=2, angle=0, color=color_tuple, fill=None)
                ax.add_patch(tail_patch)

            mcf_fuse_rect = tracker.tail_rect_list[-1]
            [x, y, w, h]  = mcf_fuse_rect[:4]
            # Draw fused rect in yellow.
            tracker_rect = mpatches.Rectangle(xy=[x, y], width=w, height=h,
                             color=(1,1,0,0.6), lw=2, fill=None) #color_b,g,r
            ax.add_patch(tracker_rect)
            # mark out the stable tracker in Magenta:
            if tracker.tracked_Frames >= tracker.integrated_frames:
                target_circle = mpatches.Circle((x+w/2, y+h/2), max(int(w), int(h)),
                                                lw=2., linestyle='-', color=(1, 0, 1, 1),
                                                fill=False, alpha=0.8)
                ax.text(x + w / 2, y + h / 2, 'T%d' % tracker.tid, fontsize=12, color=(1, 1, 1, 1),fontweight='bold')
                ax.add_patch(target_circle)
        # self.fig.savefig('/Users/yizhou/paper/report/2021南京导航雷达论坛/image_sequences/SNR12/%02d' % frame_no,
        #                  bbox_inches='tight',
        #                  pad_inches=0, dpi=300)

    def prun_overlapped_trajecotry(self, mcf_tracker_list, last_m = 9, dist_thresh=8):
        '''
        Combine the trajecotry with near centroid in the current frame.
        When two trajectory is closer than @dist_thresh in @last_m frames
        :param mcf_tracker_list:
        :return:
        '''
        N = len(mcf_tracker_list)
        remove_status_arr = np.zeros(N)#initial the remove status array as 0, means NOT remove.
        #two iteration loops to compare all pairs of trackers in the tracker_list, delete those get overlapped trajectories.
        #For each tracker, in all its overlapped trackers, the one with longest tailer is kept, others are deleted.
        for i in range(N):
            ti = mcf_tracker_list[i]
            relative_ids = [i]
            relative_lens= [len(ti.tail_rect_list)]
            for j in range(i+1, N):
                tj =mcf_tracker_list[j]
                if remove_status_arr[j] == True : # j th tracker is already deleted, omit this iteration
                    continue
                else:
                    bremove=self.trajectory_overlap(last_m, ti.tail_rect_list, tj.tail_rect_list, dist_thresh)
                    if bremove==True:
                        remove_status_arr[j] = True
                        relative_ids.append(j)
                        relative_lens.append(len(tj.tail_rect_list))
            if len(relative_ids)>1:#if overlapped traj detected, ti is marked as removable.
                remove_status_arr[i] = True
            #keep the maximum tracker length.
            remove_status_arr[np.argmax(relative_lens)] = False

        #remove the tracker with overlapped trajectory.
        for id, mcfTracker in enumerate(list(mcf_tracker_list)): #copy list to maintain the index, in the case of list.remove happens.
            if remove_status_arr[id] == True:
                print('MCF TRACKER %d is deleted, due to overlapped trajectory.'% mcfTracker.tid)
                mcf_tracker_list.remove(mcfTracker)

        return mcf_tracker_list


    def trajectory_overlap(self, number_of_latest_frames, ref_rect_list, test_rect_list, dist_thresh = 8 ):
        '''
        Compare last M frame's average distance of centroid of two trajectory.
        The last M results are concerned only.
        '''
        M = number_of_latest_frames
        if len(ref_rect_list)<M or len(test_rect_list)<M: #Traj is not long enough.
            return False
        trk_rect_arr = np.array(ref_rect_list[-M:])
        gt_rect_arr  = np.array(test_rect_list[-M:])
        dcx = (trk_rect_arr[:, 0] + trk_rect_arr[:, 2] / 2) - (gt_rect_arr[:, 0] + gt_rect_arr[:, 2] / 2)
        dcy = (trk_rect_arr[:, 1] + trk_rect_arr[:, 3] / 2) - (gt_rect_arr[:, 1] + gt_rect_arr[:, 3] / 2)
        epos_dist = np.sqrt(dcx ** 2 + dcy ** 2)
        if np.mean(epos_dist) < dist_thresh:
            return True # two traj is overlapped
        else:
            return False





    def record(self, swerling_type, snr):
        # Measure the precision between the target_trajectory and the gt_trajectory.
        # reform gt_dict {'fid0':{'tid1'[cx,cy,w,h,theta], 'tid2':[...]}, ..., 'fidn':[...]},
        # based on the fid, each frame get multiple tids.
        # to targets_gt_dict{'tid1':{'fid0':[rect], ..., 'fidn':[rect]}, 'tid2':{}, ..., 'tidn':{}},
        # based on the tid, arcoss the frames.
        targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(self.gt_trajecotry_dict)

        # evaluate the tracking results 1st version
        match_dict = eval_model.match_trajectory(targets_gt_dict, self.target_trajectory_dict)
        if self.bVerbose:
            print(match_dict)

        # # write the gt_file
        # for target_name in targets_gt_dict:
        #     with open('/Users/yizhou/code/taes2021/results/taes20_gt_%s_rect.pickle'%target_name, 'wb') as f:
        #         pickle.dump(targets_gt_dict[target_name], f, protocol=pickle.HIGHEST_PROTOCOL)
        #     # write the matched tracker's tracking results.
        #     fragmentation_id = 1 # in case there are multiple fragmentations.
        #     for target_name in match_dict:
        #         for track_id in match_dict[target_name]:
        #             file_name = '/Users/yizhou/code/taes2021/results/taes20_CF-TBD_%s_rect_frag%d.pickle' %(target_name,fragmentation_id)
        #             fragmentation_id += 1
        #             with open(file_name, 'wb') as f:
        #                 pickle.dump(target_trajectory_dict[track_id], f, protocol=pickle.HIGHEST_PROTOCOL)

        # evaluate the tracking results adding the rate of hit (roh), track fragmentation(tf), width error and height error.
        precision_dict, false_alarm_acc, overall_roh_dict = eval_model.measure_trajectory_precesion(targets_gt_dict,
                                                                                                    self.target_trajectory_dict)
        # compute pfa and pd via track_trajectory and gt_trajectory
        frame_track_trajectory = sp_model.convert_target_trajectory_to_frame_trajectory(self.target_trajectory_dict)
        fid_trk_list, self.pfa_trk_list, self.pd_trk_list = sp_model.get_pfa_pd_via_trajectory_rrect_v2(
            frame_track_trajectory, self.gt_trajecotry_dict,precision_dict, self.img_h, self.img_w, bshrink_tbox=True)
        print('MCF-TBD IntFrames = %d, Gamma= %.2f' % (self.integrated_frames, self.integrated_merits_gamma))
        print('CFAR AVE Pfa_cfar %.5f, Cfar AVE Pd %.5f' % (np.mean(self.pfa_cfar_list), np.mean(self.pd_cfar_list)))
        print('Seg  AVE Pfa_Seg  %.5f, Seg  AVE Pd %.5f' % (np.mean(self.pfa_seg_list), np.mean(self.pd_seg_list)))
        print('Trck AVE Pfa_trck %.5f, Trk  AVE Pd %.5f' % (np.mean(self.pfa_trk_list), np.mean(self.pd_trk_list)))
        print('False Alarm Reduction Gain %.2fdb' % (
                    10 * np.log10(np.mean(self.pfa_seg_list) / np.mean(self.pfa_trk_list))))
        print('Detection Increase Gain    %.2fdb' % (
                    10 * np.log10(np.mean(self.pd_trk_list) / np.mean(self.pd_seg_list))))

        nframes = len(self.gt_trajecotry_dict.keys())
        _, _, res_table = eval_model.print_metrics(precision_dict, false_alarm_acc, self.img_w, self.img_h, nframes,
                                                   overall_roh_dict)


        if self.bVerbose:
            eval_model.draw_track_traj(self.img_w, self.img_h, targets_gt_dict, self.target_trajectory_dict, precision_dict)
            eval_model.draw_rmse(precision_dict)
            #draw pfa vs fid, and pd vs fid.
            fig_pfa, pfaax = plt.subplots()
            fig_pd, pdax = plt.subplots()
            pfaax.plot(self.fid_list, self.pfa_cfar_list, color=(0.5, 0, 0),
                       label='pfa_cfar[%.5f]' % np.mean(self.pfa_cfar_list))
            pfaax.plot(self.fid_list, self.pfa_seg_list, color=(0.5, 0.5, 0),
                       label='pfa_seg[%.5f]' % np.mean(self.pfa_seg_list))
            pfaax.plot(fid_trk_list, self.pfa_trk_list, color=(1, 0, 0),
                       label='pfa_trk [%.5f]' % np.mean(self.pfa_trk_list))
            pfaax.legend()
            pdax.plot(self.fid_list, self.pd_seg_list, color=(0, 0.5, 0),
                      label='pd_seg[%2.2f]' % np.mean(self.pd_seg_list))
            pdax.plot(fid_trk_list, self.pd_trk_list, color=(0, 1, 0),
                      label='pd_trk[%2.2f]-pfa_trk[%.1E]' % (np.mean(self.pd_trk_list), np.mean(self.pfa_trk_list)))
            pdax.legend()

        pfa_gain = 10 * np.log10(np.mean(self.pfa_seg_list) / np.mean(self.pfa_trk_list))
        pd_gain  = 10 * np.log10(np.mean(self.pd_trk_list)  / np.mean(self.pd_seg_list))

        #record which writes to the 'csv' file.
        record = [swerling_type, snr, np.mean(self.time_counter_list), np.mean(self.pfa_cfar_list), np.mean(self.pfa_seg_list),
                  np.mean(self.pfa_trk_list), pfa_gain,
                  np.mean(self.pd_cfar_list), np.mean(self.pd_seg_list), np.mean(self.pd_trk_list), pd_gain,
                  self.integrated_frames, self.integrated_merits_gamma, cfar_seg_kval, cfar_ref, cfar_guide,
                  res_table['ave_res']['ave_epos'], res_table['ave_res']['ave_iou'], res_table['ave_res']['roh'],
                  res_table['ave_res']['ntf'],
                  res_table['victor']['ave_epos'], res_table['victor']['ave_iou'], res_table['victor']['roh'],
                  res_table['victor']['ntf'],
                  res_table['amelia']['ave_epos'], res_table['amelia']['ave_iou'], res_table['amelia']['roh'],
                  res_table['amelia']['ntf'],
                  res_table['urich']['ave_epos'], res_table['urich']['ave_iou'], res_table['urich']['roh'],
                  res_table['urich']['ntf']]
        record_txt = []
        for rec in record:
            record_txt.append(str(rec))

        self.csvwriter.writerow(record_txt)
        self.csvfile.flush()

    def activate(self,frame, frame_no, sw_type, snr):
        '''
        activate works in 3 conditions,
        1st frame(initial),
        intermediate frame(update and draw), and
        last frame(record and draw).
        :param frame:
        :param frame_no:
        :param sw_type:
        :param snr:
        :return:
        '''
        self.fid_list.append(frame_no)
        uframe = uti.frame_normalize(frame) * 255
        uframe = uframe.astype(np.uint8)

        time_cost_per_frame = time.perf_counter()
        blob_bb_list, bin_image = cfar_model.segmentation(uframe, lbp_contrast_select=False, kval=cfar_seg_kval,
                                                          least_wh=(3, 3),
                                                          nref=cfar_ref, mguide=cfar_guide, min_area=cfar_seg_min_area,
                                                          max_area=cfar_seg_max_area)
        # if len(blob_bb_list)==0:#This should not happen for the first frame.
        #     print('No blob')
        if frame_no ==self.init_frame:
            self.clear_info_list() #refresh the information list when a new start frame_no comes.
            self.fid_list= [frame_no]

            ##using the gt to initial the tracker for test the tracking ability.
            # frame_key = '%02d'%frame_no
            # target_rects = []
            # for tname in self.gt_rotate_rect_trajectory_dict[frame_key]:
            #     rect = self.gt_rotate_rect_trajectory_dict[frame_key][tname]
            #     rect = np.int0(rect[:4])
            #     target_rects.append(rect)
            # for rid, tbox in enumerate(target_rects):
            #     # initialize mcfTracker for each blob.
            #     mcfTracker = mcf_model.MCF_Tracker(frame, frame_no, tbox, rid,
            #                                        self.integrated_frames, kernel_sigma=self.ksigma)
            #     self.mcf_tracker_list.append(mcfTracker)

            #Init tracker by segmentation.
            for bid, blob_bb in enumerate(blob_bb_list):
                # initialize mcfTracker for each blob.
                mcfTracker = mcf_model.MCF_Tracker(frame, frame_no, blob_bb, bid,
                                                   self.integrated_frames, kernel_sigma=self.ksigma)
                self.mcf_tracker_list.append(mcfTracker)

        if frame_no > self.init_frame:
            voted_blob_id = []
            assigned_tid  = [0]
            # each blob associates with a mcf_tracker, each mcf_tracker holds multiple KCF trackers.
            for mcfTracker in self.mcf_tracker_list:
                mcf_fuse_rect, psr = mcfTracker.update(frame, frame_no, blob_bb_list)
                if mcfTracker.voted_blob_id is not None:
                    voted_blob_id.append(mcfTracker.voted_blob_id)
                assigned_tid.append(mcfTracker.tid)

            # Delete mcf_tracker which has low ave_psr value and low integrated_merits_gamma
            del_mcf_via_psr_nums   = 0
            del_mcf_via_gamma_nums = 0

            # #prun overlapped trajectory!
            # if len(self.fid_list)>self.integrated_frames:
            #     self.prun_overlapped_trajecotry(self.mcf_tracker_list, last_m=9, dist_thresh=8)

            #Confirm qualified trackers and delete the losers.
            for mcfTracker in list(self.mcf_tracker_list):
                if (mcfTracker.tracked_Frames >= mcfTracker.integrated_frames) \
                        and (mcfTracker.integrated_merits>=(self.integrated_merits_gamma)):
                    #Confirm trajectory here, save it in the target_trajectory_dict.
                    self.target_trajectory_dict[mcfTracker.tid] = mcfTracker.get_target_trajectory()
                    #Print detail information for debug
                    if(self.bVerbose):
                        tbox = mcfTracker.tail_rect_list[-1]
                        tx, ty, tw, th = tbox[:4]
                        target_roi = frame[ty:ty + th, tx:tx + tw]
                        region_snr = 10 * np.log10(np.mean(target_roi ** 2) / 2)  # simulated clutter's average power is 2.
                        print('MKCF_Tid %3d, Average PSR %2.2f   , Average_life of KCF %3.1f  , region_snr %.2f'
                              ' Merit \u03BB_k %2.4f , Integrated psrs %.2f' %
                              (mcfTracker.tid, mcfTracker.ave_psr, mcfTracker.ave_life, region_snr,
                               mcfTracker.integrated_merits, mcfTracker.integrated_psrs))
                # bool status for weak average psr in the starting stages.
                b_weak_ave_psr = (mcfTracker.tracked_Frames < mcfTracker.integrated_frames and mcfTracker.ave_psr < mcfTracker.votePsrThreash)
                # bool status for low integrated merit scores after tracker has integrated enough frames.
                b_low_int_merit= (mcfTracker.tracked_Frames >= mcfTracker.integrated_frames and mcfTracker.integrated_merits<self.integrated_merits_gamma)
                if b_weak_ave_psr or b_low_int_merit: # delete low quality trackers to release mcf_tbd burden
                        #and mcfTracker.integrated_merits < self.integrated_merits_gamma):
                            # save the qualified long_term tracker in the target_trajectory_dict, before delete the tracker
                            # confirmation the trajectory based on two condition: tracked frames is long enough
                            # and the integrated merits is bigger than gamma
                            self.mcf_tracker_list.remove(mcfTracker)
                            assigned_tid.remove(mcfTracker.tid)
                            if b_weak_ave_psr:
                                del_mcf_via_psr_nums   += 1
                            if b_low_int_merit:
                                del_mcf_via_gamma_nums +=1

            # # add new tracker for the new segmented blobs.
            # assigned_tid.sort()
            ini_mcf_nums = 0
            # # initialize unlabelled blob for new tracker
            for bid, blob_bb in enumerate(blob_bb_list):
                # for mcfTracker in self.mcf_tracker_list:
                #     tbox = mcfTracker.fuse_rect
                #     # if uti.intersection_rect(blob_bb, tbox)>0:
                #     #     print(mcfTracker.tid, uti.intersection_rect(blob_bb, tbox))
                #     if uti.intersection_area(blob_bb, tbox)/(blob_bb[2]*blob_bb[3]) > 0.9:
                #         #exist tbox already contains the blob, no initialization happens
                #         #print('MKCF tid %d gets overlapped new blob!'%mcfTracker.tid)
                #         voted_blob_id.append(bid)
                #Fresh blobs which are not used for component KCF trackers are used to initialize new trackers.
                if bid not in voted_blob_id:
                    tid = assigned_tid[-1] + 1  # new target id
                    # Initialized with new target id
                    # Enlarge the segmented blobs in MCF_TRACKER
                    new_mcf_tracker = mcf_model.MCF_Tracker(frame, frame_no, blob_bb, tid,
                                                            self.integrated_frames,kernel_sigma=self.ksigma)
                    self.mcf_tracker_list.append(new_mcf_tracker)
                    assigned_tid.append(tid)
                    ini_mcf_nums += 1

        time_cost_per_frame = time.perf_counter() - time_cost_per_frame
        self.time_counter_list.append(time_cost_per_frame)

        pfa_cfar, pd_cfar, pfa_seg, pd_seg = \
            sp_model.get_pfa_pd_via_cfar_rrect(bin_image, self.gt_rotate_rect_trajectory_dict['%02d' % frame_no],
                                               blob_bb_list)
        self.pfa_cfar_list.append(pfa_cfar)
        self.pd_cfar_list.append(pd_cfar)
        self.pfa_seg_list.append(pfa_seg)
        self.pd_seg_list.append(pd_seg)

        if frame_no == self.last_frame:
           #write the tracking results as a record text in csvfile.
           self.record(sw_type, snr)
        if self.bVerbose:
            self.draw_bounding_box(frame, frame_no, self.ax, blob_bb_list, self.mcf_tracker_list,snr)

            plt.pause(0.01)
            #plt.waitforbuttonpress()
            print('Frame %02d maintains %4d mcf_trackers' % (frame_no, len(self.mcf_tracker_list)))
            if frame_no>self.init_frame:
                print('Frame %02d delete %04d weak mcf_trackers with less psr, delete %04d trackers with less gamma, while init %04d new mcf_trackers'
                    % (frame_no, del_mcf_via_psr_nums, del_mcf_via_gamma_nums, ini_mcf_nums))
            print('pfa_cfar %.5f pd_cfar %.2f pfa_seg %.5f pd_seg %.2f' % (pfa_cfar, pd_cfar, pfa_seg, pd_seg))



class DP_TBD_SIM(): #Grossi_DP_TBD Tracker and LELR_DP_TBD are implemented for simulated situations.
    def __init__(self, dp_tbd_type, img_w, img_h, start_fid, end_fid,
                       integrated_frames, integrated_merits_gamma,
                       bVerbose=False):
        self.dp_tbd_type = dp_tbd_type  #define the dp_tbd tracker's type: 'GROSSI' and 'LELR', with different merits(score) function
        #simulated image size
        self.img_w                   = img_w
        self.img_h                   = img_h
        #parameters setting for MCF_TBD
        self.integrated_frames       = integrated_frames
        self.integrated_merits_gamma = integrated_merits_gamma
        #show debug information or not, show trajectory figures or not
        self.bVerbose                = bVerbose
        self.number_of_alarm_list    = []      #number of alarms per frame
        self.size_of_alarm_list      = []      #size (wxh) of extended alarm per frame
        if (self.bVerbose):
            self.fig = plt.figure(figsize=(7, 7))
            self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (800, 0))
            # New axis over the whole figure, no frame and a 1:1 aspect ratio
            self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        #set the start and end frame_no.
        self.init_frame              = start_fid #start frame no.
        self.last_frame              = end_fid   # the last frame no.
        # snr   = 0         # snr-ksigma(20, 0.12) (10, 0.045) (3, 0.023) (0, 0.023)
        self.fid_list = []
        # name of csv file
        date = datetime.now()
        if dp_tbd_type == 'GROSSI':
            filename = "/Users/yizhou/code/taes2021/results/GROSSI_TBD_SIMULATE_RECORDS_%02d_%02d.csv" \
                       %(date.month, date.day)
            #self.dp_tbd = gross_tbd_model.DP_TBD_Grossi(P=4, Q=3, L=self.integrated_frames,
            #                                             gamma2=self.integrated_merits_gamma)
        if dp_tbd_type == 'LELR':
            filename = "/Users/yizhou/code/taes2021/results/LELR_TBD_SIMULATE_RECORDS_%02d_%02d.csv" \
                       %(date.month, date.day)
            #self.dp_tbd = lelr_tbd_model.DP_TBD_LELR_Grossi(P=4, Q=3, L=self.integrated_frames,
            #                                                gamma2=self.integrated_merits_gamma, swerling_type=0)

        # writing to csv file
        self.csvfile = open(filename, 'w')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(record_fields)

    def set_new_record_file(self, iter):
        '''
        each iteration gets one record file.
        :param iter:
        :return:
        '''
        self.csvfile.close()
        date = datetime.now()
        date = datetime.now()
        if self.dp_tbd_type == 'GROSSI':
            filename = "/Users/yizhou/code/taes2021/results/GROSSI_TBD_SIMULATE_RECORDS_%02d_%02d_%02d.csv" \
                       %(date.month, date.day, iter)
            #self.dp_tbd = gross_tbd_model.DP_TBD_Grossi(P=4, Q=3, L=self.integrated_frames,
            #                                             gamma2=self.integrated_merits_gamma)
        if self.dp_tbd_type == 'LELR':
            filename = "/Users/yizhou/code/taes2021/results/LELR_TBD_SIMULATE_RECORDS_%02d_%02d_%02d.csv" \
                       %(date.month, date.day,iter)
        # writing to csv file
        self.csvfile   = open(filename, 'w')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(record_fields)

    def set_gt_dict(self,gt_rotate_rect_trajectory_dict, gt_trajecotry_dict):
        '''
        Assign the ground-truth dict information.
        :param gt_rotate_rect_trajectory_dict:
        :param gt_trajecotry_dict:
        :return:
        '''
        self.gt_rotate_rect_trajectory_dict = gt_rotate_rect_trajectory_dict
        self.gt_trajecotry_dict             = gt_trajecotry_dict

    def clear_info_list(self):
        #clear the information list
        msim_model.local_snrs        = []  # clear the  local snrs records
        msim_model.global_snrs       = []  # clear the  msim_model's global snr records.

        self.fid_list                = []
        self.pfa_cfar_list           = []
        self.pfa_seg_list            = []
        self.pfa_trk_list            = []

        self.pd_cfar_list            = []
        self.pd_seg_list             = []
        self.pd_trk_list             = []

        self.mcf_tracker_list        = []
        self.time_counter_list       = []
        # save the qualified [mctracker.tracked_Frames > nums_thresh] target trajectory in the dict.
        # This dict's format is the same as the gt_dict in 'simulate_clutter_target_*.py'
        # {'target_name':{frame_no:[rect_x, y, w,h]}}
        self.target_trajectory_dict  = {}

    def record(self, swerling_type, snr):

        self.dp_tbd.transfer_tau_to_trajectory_dict(self.dp_tbd.target_tau_dict)


        # Measure the precision between the target_trajectory and the gt_trajectory.
        # reform gt_dict {'fid0':{'tid1'[cx,cy,w,h,theta], 'tid2':[...]}, ..., 'fidn':[...]},
        # based on the fid, each frame get multiple tids.
        # to targets_gt_dict{'tid1':{'fid0':[rect], ..., 'fidn':[rect]}, 'tid2':{}, ..., 'tidn':{}},
        # based on the tid, arcoss the frames.
        targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(self.gt_trajecotry_dict)

        # evaluate the tracking results 1st version
        match_dict = eval_model.match_trajectory(targets_gt_dict, self.dp_tbd.target_trajectory_dict)
        if self.bVerbose:
            print(match_dict)

        # evaluate the tracking results adding the rate of hit (roh), track fragmentation(tf), width error and height error.
        precision_dict, false_alarm_acc, overall_roh_dict = eval_model.measure_trajectory_precesion(targets_gt_dict,
                                                                                                    self.dp_tbd.target_trajectory_dict)

        # compute pfa and pd via track_trajectory and gt_trajectory
        frame_track_trajectory = sp_model.convert_target_trajectory_to_frame_trajectory(self.dp_tbd.target_trajectory_dict, bshrink=False)
        self.fid_trk_list, self.pfa_trk_list, self.pd_trk_list = sp_model.get_pfa_pd_via_trajectory_rrect_v2(frame_track_trajectory,
                                                                                           self.gt_trajecotry_dict, precision_dict,
                                                                                           self.img_h, self.img_w, bshrink_tbox=False)
        if self.dp_tbd_type =='GROSSI':
            print('GROSSI-TBD IntFrames = %d, Gamma= %.2f'%(self.integrated_frames, self.integrated_merits_gamma))
        if self.dp_tbd_type == 'LELR':
            print('LELR-TBD IntFrames = %d, Gamma= %.5f' % (self.integrated_frames, self.integrated_merits_gamma))
        print('CFAR AVE Pfa_cfar %.5f, Cfar AVE Pd %.5f' % (np.mean(self.pfa_cfar_list), np.mean(self.pd_cfar_list)))
        print('Seg  AVE Pfa_Seg  %.5f, Seg  AVE Pd %.5f' % (np.mean(self.pfa_seg_list),  np.mean(self.pd_seg_list)))
        print('Trck AVE Pfa_trck %.5f, Trk  AVE Pd %.5f' % (np.mean(self.pfa_trk_list),  np.mean(self.pd_trk_list)))
        print('False Alarm Reduction Gain %.2fdb' % (10 * np.log10(np.mean(self.pfa_seg_list) / np.mean(self.pfa_trk_list))))
        print('Detection Increase Gain    %.2fdb' % (10 * np.log10(np.mean(self.pd_trk_list)  / np.mean(self.pd_seg_list))))

        nframes = len(self.gt_trajecotry_dict.keys())
        _, _, res_table = eval_model.print_metrics(precision_dict, false_alarm_acc, self.img_w, self.img_h, nframes,
                                                   overall_roh_dict)

        print('===Average number of blobs per frame %.2f  ===' % np.mean(self.number_of_alarm_list))
        print('===Average size of blobs in all frames %.2f===' % np.mean(self.size_of_alarm_list))


        if self.bVerbose:
            eval_model.draw_track_traj(self.img_w, self.img_h, targets_gt_dict, self.dp_tbd.target_trajectory_dict, precision_dict)
            eval_model.draw_rmse(precision_dict)
            #draw pfa vs fid, and pd vs fid.
            fig_pfa, pfaax = plt.subplots()
            fig_pd, pdax   = plt.subplots()
            pfaax.plot(self.fid_list, self.pfa_cfar_list, color=(0.5, 0, 0),
                       label='pfa_cfar[%.5f]' % np.mean(self.pfa_cfar_list))
            pfaax.plot(self.fid_list, self.pfa_seg_list, color=(0.5, 0.5, 0),
                       label='pfa_seg[%.5f]' % np.mean(self.pfa_seg_list))
            pfaax.plot(self.fid_trk_list, self.pfa_trk_list, color=(1, 0, 0),
                       label='pfa_trk [%.5f]' % np.mean(self.pfa_trk_list))
            pfaax.legend()
            pdax.plot(self.fid_list, self.pd_seg_list, color=(0, 0.5, 0),
                      label='pd_seg[%2.2f]' % np.mean(self.pd_seg_list))
            pdax.plot(self.fid_trk_list, self.pd_trk_list, color=(0, 1, 0),
                      label='pd_trk[%2.2f]-pfa_trk[%.1E]' % (np.mean(self.pd_trk_list), np.mean(self.pfa_trk_list)))
            pdax.legend()

        pfa_gain = 10 * np.log10(np.mean(self.pfa_seg_list) / np.mean(self.pfa_trk_list))
        pd_gain  = 10 * np.log10(np.mean(self.pd_trk_list)  / np.mean(self.pd_seg_list))

        #record which writes to the 'csv' file.
        record = [swerling_type, snr, np.mean(self.time_counter_list), np.mean(self.pfa_cfar_list), np.mean(self.pfa_seg_list),
                  np.mean(self.pfa_trk_list), pfa_gain,
                  np.mean(self.pd_cfar_list), np.mean(self.pd_seg_list), np.mean(self.pd_trk_list), pd_gain,
                  self.integrated_frames, self.integrated_merits_gamma, cfar_seg_kval, cfar_ref, cfar_guide,
                  res_table['ave_res']['ave_epos'], res_table['ave_res']['ave_iou'], res_table['ave_res']['roh'],
                  res_table['ave_res']['ntf'],
                  res_table['victor']['ave_epos'], res_table['victor']['ave_iou'], res_table['victor']['roh'],
                  res_table['victor']['ntf'],
                  res_table['amelia']['ave_epos'], res_table['amelia']['ave_iou'], res_table['amelia']['roh'],
                  res_table['amelia']['ntf'],
                  res_table['urich']['ave_epos'], res_table['urich']['ave_iou'], res_table['urich']['roh'],
                  res_table['urich']['ntf']]
        record_txt = []
        for rec in record:
            record_txt.append(str(rec))

        self.csvwriter.writerow(record_txt)
        self.csvfile.flush()

    def activate(self,frame, frame_no, sw_type, snr):
        '''
        activate works in 3 conditions,
        1st frame(initial),
        intermediate frame(update and draw), and
        last frame(record and draw).
        :param frame:
        :param frame_no:
        :param sw_type:
        :param snr:
        :return:
        '''
        self.fid_list.append(frame_no)

        time_cost_per_frame = time.perf_counter()
        blob_bb_list, bin_image = cfar_model.segmentation(frame, lbp_contrast_select=False, kval=cfar_seg_kval,
                                                          least_wh=(3, 3),
                                                          nref=cfar_ref, mguide=cfar_guide, min_area=cfar_seg_min_area,
                                                          max_area=cfar_seg_max_area)

        self.number_of_alarm_list.append(len(blob_bb_list))
        bsize = 0
        for blob in blob_bb_list:
            bsize += blob[2]*blob[3]
        ave_size = bsize/len(blob_bb_list)
        self.size_of_alarm_list.append(ave_size)

        ##add rotated rectangle list in wtsa_tbd_model
        # blob_bb_list, blob_rb_list, bin_image = wtsa_tbd_model.segmentation(frame, lbp_contrast_select=False, kval=cfar_seg_kval,
        #                                                   least_wh=(3, 3),
        #                                                   nref=cfar_ref, mguide=cfar_guide, min_area=cfar_seg_min_area,
        #                                                   max_area=cfar_seg_max_area)
        # #enlarge segmented blob twice, to increase the detection rate.
        # blob_bb_list = []
        # for blob in blob_sm_list:
        #     bx,by,bw,bh = blob[:4]
        #     cx,cy = bx+bw/2, by+bh/2
        #     nbw, nbh = 2*bw, 2*bh
        #     nbb = np.int0([max(0, cx-nbw/2), max(0, cy-nbh/2), min(self.img_w, nbw), min(self.img_h,nbh)])
        #     blob_bb_list.append(nbb)

        if len(blob_bb_list)==0:#This should not happen for the first frame.
             print('lower to cfar_seg_kval to get non-zero blobs!')
        if frame_no ==self.init_frame:
            self.clear_info_list() #refresh the information list when a new start frame_no comes.
            self.fid_list= [frame_no]
            # if self.dp_tbd_type=='LELR':
            #     self.dp_tbd.swerling_type = sw_type # Reset the dp_tbd's Swerling type
            if self.dp_tbd_type == 'GROSSI':
                filename = "/Users/yizhou/code/taes2021/results/GROSSI_TBD_SIMULATE.csv"
                self.dp_tbd = gross_tbd_model.DP_TBD_Grossi(P=4, Q=3, L=self.integrated_frames,
                                                            gamma2=self.integrated_merits_gamma)
            if self.dp_tbd_type == 'LELR':
                filename = "/Users/yizhou/code/taes2021/results/LELR_TBD_SIMULATE.csv"
                self.dp_tbd = lelr_tbd_model.DP_TBD_LELR_Grossi(P=4, Q=3, L=self.integrated_frames,
                                                                gamma2=self.integrated_merits_gamma, swerling_type=sw_type)
                # self.dp_tbd = wtsa_tbd_model.DP_TBD_LELR_Grossi(P=4, Q=3, L=self.integrated_frames,
                #                                                 gamma2=self.integrated_merits_gamma, swerling_type=sw_type)

        #if self.dp_tbd_type =='GROSSI':
        self.dp_tbd.generate_nodes(frame, frame_no, blob_bb_list, timestep=1)
        # if self.dp_tbd_type == 'LELR':#WTSA, using the orientated gaussian coefficients.
        #     self.dp_tbd.generate_nodes(frame, frame_no, blob_bb_list, blob_rb_list, timestep=1)
        self.dp_tbd.find_neighbour(frame_no, self.dp_tbd.nodes_dict)
        self.dp_tbd.generate_trajectory(frame_no, self.dp_tbd.nodes_dict)
        if frame_no >= (self.dp_tbd.L - 1):  # fid start from 0
            # trajectory formation in last frame's node.tau list.
            # dp_tbd.generate_trajectory(fid, dp_tbd.nodes_dict)
            # prune trajectory and confirm the trajectory
            # dp_tbd.print_nodes_dict(dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(dp_tbd.nodes_dict)
            prun_nodes_dict = self.dp_tbd.prun_trajectory(self.dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(prun_nodes_dict)
            self.dp_tbd.confirm_trajectory(frame_no, self.dp_tbd.nodes_dict, self.dp_tbd.gamma2)
        time_cost_per_frame = time.perf_counter() - time_cost_per_frame
        self.time_counter_list.append(time_cost_per_frame)


        pfa_cfar, pd_cfar, pfa_seg, pd_seg = \
            sp_model.get_pfa_pd_via_cfar_rrect(bin_image, self.gt_rotate_rect_trajectory_dict['%02d' % frame_no],
                                               blob_bb_list)
        self.pfa_cfar_list.append(pfa_cfar)
        self.pd_cfar_list.append(pd_cfar)
        self.pfa_seg_list.append(pfa_seg)
        self.pd_seg_list.append(pd_seg)
        if frame_no == self.last_frame:
           #write the tracking results as a record text in csvfile.
           self.record(sw_type, snr)
        if self.bVerbose:
            self.fig.canvas.set_window_title('frame%3d' % frame_no)
            print('pfa_cfar %.5f pd_cfar %.2f pfa_seg %.5f pd_seg %.2f' % (pfa_cfar, pd_cfar, pfa_seg, pd_seg))
            gross_tbd_model.draw_bounding_boxs(frame, blob_bb_list, self.ax, color=(0, 1, 1))
            self.dp_tbd.draw_traj(self.ax, frame_no, self.dp_tbd.target_tau_dict)
            plt.pause(0.01)



def test_cfar_seg_snr(nframes, img_w, img_h, gt_rotate_rect_trajectory_dict):
    # test real snr for the segmented target roi(which is shrinked compared to the GT target roi)
    # decide the decrease coefficient in the Gaussian Template.
    # template =  template*dec_coef in msim_model.add_gaussian_template_on_clutter()
    fig, ax = plt.subplots()
    snrs = list(range(12, -2, -1))
    line_marks =  ['o', 'v', '^', 's']
    colors     =  ['r', 'g', 'y', 'b']
    for sw_type in [0,1,3]:#[0, 1, 3]:  # , 1, 3]:
        mean_vic_snrs = []
        mean_ame_snrs = []
        mean_uri_snrs = []

        mean_peak_vic = []
        mean_peak_ame = []
        mean_peak_uri = []
        for snr in snrs:
            realsnr_vic_list = []
            realsnr_ame_list = []
            realsnr_uri_list = []
            peaksnr_vic_list = [] #conerns only the peak_snr
            peaksnr_ame_list = []
            peaksnr_uri_list = []
            width_ratio_list = []
            height_ratio_list= []

            msim_model.global_snrs = [] #clear snr record each time
            msim_model.local_snrs  = []
            for frame_no in range(nframes):
                frame = msim_model.get_frame(img_w, img_h, frame_no, snr, gt_rotate_rect_trajectory_dict, sw_type)
                blob_bb_list, bin_image = cfar_model.segmentation(frame, lbp_contrast_select=False, kval=cfar_seg_kval,
                                                                  least_wh=(3, 3),
                                                                  nref=cfar_ref, mguide=cfar_guide,
                                                                  min_area=cfar_seg_min_area,
                                                                  max_area=cfar_seg_max_area)
                frame_key = '%02d' % frame_no
                names = []
                target_rects = []
                for tname in gt_rotate_rect_trajectory_dict[frame_key]:
                    rect = gt_rotate_rect_trajectory_dict[frame_key][tname]
                    rect = np.int0(rect[:4])
                    target_rects.append(rect)
                    names.append(tname)
                for blob in blob_bb_list:
                    for id, trect in enumerate(target_rects):
                        if uti.intersection_rect(blob, trect) > 0.2:
                            tx, ty, tw, th = blob[:4]
                            roi = frame[ty:ty + th, tx:tx + tw]
                            real_snr = 10 * np.log10(max(np.mean((roi-np.sqrt(2)) ** 2), np.spacing(1)) / 2)#(signal-clutter
                            peak     = max( np.max(roi-np.sqrt(2)), np.spacing(1))
                            peak_snr = 10 * np.log10(peak**2/2)
                            if names[id] == 'victor':
                                realsnr_vic_list.append(real_snr)
                                peaksnr_vic_list.append(peak_snr)
                            if names[id] == 'amelia':
                                realsnr_ame_list.append(real_snr)
                                peaksnr_ame_list.append(peak_snr)
                            if names[id] == 'urich':
                                realsnr_uri_list.append(real_snr)
                                peaksnr_uri_list.append(peak_snr)
                            # print('template w-h: %2d-%2d  , segblob w-h: %2d-%2d, shrink (%.2f-%.2f)'%
                            #        (trect[2], trect[3], tw, th, trect[2]/tw, trect[3]/th))
                            width_ratio_list.append(trect[2] / tw)
                            height_ratio_list.append(trect[3]/ th)
                #             print('Frame %d, %10s sw %d, snr %2d,extented snr: %.2f, peak snr %.2f' % (frame_no, names[id], sw_type, snr, real_snr, peak_snr))
                #             patch = mpatches.Rectangle(xy=[tx, ty], width=tw, height=th, color=(0, 1, 1, 0.8), lw=2,fill=None)  # color r,g,b, blob in light blue.
                #             ax.add_artist(patch)
                # ax.imshow(frame)
                # plt.pause(0.001)
                # ax.clear()
            print('%10s sw %d, snr %2d, extended real-snr %.2f' % ('victor', sw_type, snr, np.mean(realsnr_vic_list)))
            print('%10s sw %d, snr %2d, extended real-snr %.2f' % ('amelia', sw_type, snr, np.mean(realsnr_ame_list)))
            print('%10s sw %d, snr %2d, extended real-snr %.2f' % ('uric',   sw_type, snr, np.mean(realsnr_uri_list)))

            print('%10s sw %d, snr %2d, peak     real-snr %.2f' % ('victor', sw_type, snr, np.mean(peaksnr_vic_list)))
            print('%10s sw %d, snr %2d, peak     real-snr %.2f' % ('amelia', sw_type, snr, np.mean(peaksnr_ame_list)))
            print('%10s sw %d, snr %2d, peak     real-snr %.2f' % ('uric',   sw_type, snr, np.mean(peaksnr_uri_list)))

            print('sw %d, snr %2d wr-hr: %.2f-%.2f' % (sw_type, snr, np.mean(width_ratio_list), np.mean(height_ratio_list)))
            mean_vic_snrs.append(np.mean(realsnr_vic_list))
            mean_ame_snrs.append(np.mean(realsnr_ame_list))
            mean_uri_snrs.append(np.mean(realsnr_uri_list))
            mean_peak_vic.append(np.mean(peaksnr_vic_list))
            mean_peak_ame.append(np.mean(peaksnr_ame_list))
            mean_peak_uri.append(np.mean(peaksnr_uri_list))
        plt.plot(snrs, mean_vic_snrs, color=colors[sw_type], label='victor_region sw %d'%sw_type, marker=line_marks[sw_type])
        plt.plot(snrs, mean_ame_snrs, color=colors[sw_type], label='amelia_region sw %d'%sw_type, marker=line_marks[sw_type])
        plt.plot(snrs, mean_uri_snrs, color=colors[sw_type], label='urich_region  sw %d'%sw_type, marker=line_marks[sw_type])

        plt.plot(snrs, mean_peak_vic, color=colors[sw_type], linestyle='-.', label='victor_peak sw %d'%sw_type, marker=line_marks[sw_type])
        plt.plot(snrs, mean_peak_ame, color=colors[sw_type], linestyle='-.', label='amelia_peak sw %d'%sw_type, marker=line_marks[sw_type])
        plt.plot(snrs, mean_peak_uri, color=colors[sw_type], linestyle='-.', label='urich_peak  sw %d'%sw_type, marker=line_marks[sw_type])

    plt.plot(snrs[::-1],snrs[::-1],'--', color='black')
    ax.set_xlim((-2,13))
    ax.set_title('Target Ideal SNR Vs. In Clutter SNR')
    plt.legend()
    #print(msim_model.dec)
    plt.pause(0.01)
    plt.show()

def test_trackers_on_simulation():
    '''
    Test the 3 trackers on the same simulation.
    :return:
    '''

    gt_rotate_rect_trajectory_dict = msim_model.multiple_extended_targets_in_clutter()
    gt_trajecotry_dict = sp_model.convert_rotateRect_to_Rect_trajectory_dict(gt_rotate_rect_trajectory_dict)
    nframes = len(gt_trajecotry_dict.keys())  # tested frames.
    img_w             = 300
    img_h             = 300
    # cfar_seg_max_area = 30000  # 200+ for inesa, 100  for taes20_sim
    # cfar_seg_min_area = 90  # 200+ for titan, 100  for TriFalo
    # cfar_ref          = 16 * 2  # Titan 20,  TriFalo 10.
    # cfar_guide        = 8  * 2  # Titan 10,  TriFalo 8.
    #
    # cfar_seg_kval   = 1.8  # 1.3


    ksigma            = 0.015#0.015
    start_frameno     = 0
    end_frameno       = nframes-1

    # mcf_gamma         = 2.1 #integrated gamma for MCF-TBD
    # mcf_int_frames    = 9

    mcf_gamma         = 0.7 #integrated gamma for MCF-TBD
    mcf_int_frames    = 3

    gross_int_frames  = 3#9
    gross_gamma       = 6.67#20

    lelr_int_frames   = 3#9
    lelr_gamma        = 0.01/3#0.01

    test_snr          = 12
    swerling_type     = 0

    # test_cfar_seg_snr(nframes, img_w, img_h, gt_rotate_rect_trajectory_dict)
    # # print()
    # return

    mcf_sim = MCF_TBD_SIM(img_w, img_h, start_frameno, end_frameno,
                          integrated_frames=mcf_int_frames, integrated_merits_gamma=mcf_gamma,
                          ksigma=ksigma, bVerbose=False)
    mcf_sim.set_gt_dict(gt_rotate_rect_trajectory_dict, gt_trajecotry_dict)
    # single-round test for the mcftracker.
    # for frame_no in range(nframes):
    #     frame = msim_model.get_frame(img_w, img_h, frame_no, test_snr, gt_rotate_rect_trajectory_dict, swerling_type)
    #     mcf_sim.activate(frame, frame_no, sw_type=swerling_type, snr=test_snr)

    gross_sim = DP_TBD_SIM('GROSSI',img_w, img_h, start_frameno, end_frameno, integrated_frames=gross_int_frames,
                               integrated_merits_gamma=gross_gamma, bVerbose=False)
    gross_sim.set_gt_dict(gt_rotate_rect_trajectory_dict, gt_trajecotry_dict)
    # # single-round test for the GROSS_TBD.
    # for frame_no in range(nframes):
    #     frame = msim_model.get_frame(img_w, img_h, frame_no, test_snr, gt_rotate_rect_trajectory_dict, swerling_type)
    #     gross_sim.activate(frame, frame_no, sw_type=swerling_type, snr=test_snr)


    # if swerling_type==3:
    # lelr_gamma = 0.01
    lelr_sim = DP_TBD_SIM('LELR',img_w, img_h, start_frameno, end_frameno, integrated_frames=lelr_int_frames,
                               integrated_merits_gamma=lelr_gamma, bVerbose=False)
    lelr_sim.set_gt_dict(gt_rotate_rect_trajectory_dict, gt_trajecotry_dict)
    # # single-round test for the GROSS_TBD.
    # for frame_no in range(nframes):
    #     frame = msim_model.get_frame(img_w, img_h, frame_no, test_snr, gt_rotate_rect_trajectory_dict)
    #     lelr_sim.activate(frame, frame_no, sw_type=swerling_type, snr=test_snr)

    # plt.show()
    # return
    #repeat 10 times for test
    for iter in range(10):
        print('===Round %2d Testing==\n'%iter)
        mcf_sim.set_new_record_file(iter)
        gross_sim.set_new_record_file(iter)
        lelr_sim.set_new_record_file(iter)
        snrs = list(range(12, -2, -1))  # [12, 11, ..., -1, -2]
        for sw_type in [0, 1, 3]:#[0, 1, 3]:
            for snr in snrs:
                print('---swerling type is %d, snr is %d---'%(sw_type, snr))
                tcost = time.perf_counter()
                for frame_no in range(nframes):
                    # if snr > 5:
                    #     mcf_sim.ksigma = 0.03
                    # else:
                    mcf_sim.ksigma = 0.015
                    frame = msim_model.get_frame(img_w, img_h, frame_no, snr, gt_rotate_rect_trajectory_dict, sw_type)
                    mcf_sim.activate(frame, frame_no, sw_type, snr)
                    gross_sim.activate(frame, frame_no, sw_type, snr)
                    lelr_sim.activate(frame, frame_no, sw_type, snr)
                print('Go through all frames for 3 trackers cost %.2f seconds'%(time.perf_counter()-tcost))

if __name__=='__main__':
    test_trackers_on_simulation()
    print('ALL TEST COMPLETED!')