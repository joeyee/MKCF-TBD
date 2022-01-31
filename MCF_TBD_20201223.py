'''
Multiple Correlated Filters for object tracking.
Fusing multiple trackers for every object.
A re-vised version of 2020-06-03, This time mcf could use the mosse and kcf alternatively.
'''
import sys
sys.path.append("../segmentation/")  #for import the utility in up-directory/segmention/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cv2

from sklearn.cluster import KMeans

# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from scipy import ndimage
from skimage import transform
import pandas as pd
#import mosse_tracker_20200601           as mosse_model
import cfar_segmentation_200527         as cfar_segentation
import utilities_200611                 as uti
#import KCFtracker_Status_MotionVector   as kcf_model
import KCF_20210131                     as kcf_model   # Replace pylab with numpy

'''
First frame: 
1 - initial segmentation, initial trackers for every segmented object.
2 - Check the tracker's status to decide whether to add more trackers on the same object.
    Initial the new segmented object without trackers.
    Fuse MCF on every object, draw the trial of all MCF for every object.
'''

#mcf_figure, mcf_ax = plt.subplots(2,3) # for demonstrate the component's response matrix ( 5 at most)and fused response matrix
DETAIL_MODE = False  # True for print more information on the console output.
class MCF_Tracker():
    def __init__(self, frame, frame_no, target_rect, tid, integrated_frames=10, kernel_sigma = 1.2):

        self.maxTrckNo        = 5 #uppper limit
        self.minTrckNo        = 3 #lower  limit
        self.dynamicTrckNo    = self.minTrckNo   #Needed tracker numberbs for fusion.
        #self.votePsrThreash  = 8  # Sth in paper, psr > Sth means good tracking status,
        # should set 10  for the taes20_mtt_in_clutter
        self.votePsrThreash   = 8
        self.voteDistThreash  = 100   # near by trackers get the voting right, for inesa
        #self.voteDistThreash  = 100/8 # near by trackers get the voting right, for taes20_sim
        self.blobScoreThreash = 0.15 # IOU thresh (Oth in paper) for segmented blobs and target_rect.
        # target id is unchanged for each mcf_tracker
        self.tid     = tid
        # voted_blob_id is changed frame by frame, this id is given by the index of the voted blob.
        self.voted_blob_id = None
        # if blob is selected for initialzed a new component tracker
        self.init_blob_id  = tid
        self.tail_frameno_list = [frame_no] # add frame no for each tail_rect.
        self.tail_rect_list = [target_rect] # tail_rect_list store all the obj_bbx in each frame.
        self.fuse_rect = target_rect
        self.trckList  = []
        self.votable_tbox_list  = []

        #Tracker can use different type: mosse or kcf
        #tracker = mosse_model.mosse(frame, frame_no, target_rect)
        self.kernel_sigma = kernel_sigma
        tracker = kcf_model.KCFTracker(frame, target_rect, frame_no, kernel_opt='gk', kernel_sigma=kernel_sigma)
        self.trckList.append(tracker)
        self.tracked_Frames = 1
        self.ave_psr = tracker.psr #initial ave_psr.
        self.integrated_merits  = 0
        self.integrated_frames  = integrated_frames # default is 10
        self.merit_list = [0.2] # initial value equal to integrated_merits_gamma/integrated_frames for the initial frame
        self.psr_list   = [15]  # check   the psr varying
        self.integrated_psrs = 0
        self.ave_life   = 1
        self.star_psr   = 15  # the maximum psr of the component tracker
        self.star_peak  = 1   # the peak of Y^{i^(*)}
        self.star_life  = 1   # the tracked frame numbers for the component tracker with the max_psr.

    #'Vote one blob for new initialization, or check out the new incoming blob without trackers.'

    def update(self, frame, frame_no, blob_list):
        #update every trackers
        #computing the ave_psr for one objects
        ##
        # When the trackerlist is full, len(trackerlist) >= NumberOfTrackers,
        # Delete the tracker if it is not votable (psr < votePsrThreash).
        # Dynamically changing (+1 or -1) the NumberOfTrackers, according to the ave_psr in the trackerlist.
        #
        ##
        minpsr  = 10000
        maxpsr  = 0
        ave_psr = 0
        ave_life = 0
        self.votable_tbox_list  = []
        self.voted_blob_id = None
        self.init_blob_id  = None
        # if frame_no == 48+1 and self.tid==546:
        #     print('')
        # take the last target_rect as the initial obj_bbox
        obj_bbox = self.tail_rect_list[-1]
        for i, tracker in enumerate(self.trckList):
            # upate the tracking results for the new frame
            tbox, psr, ypeak = tracker.update(frame, frame_no)
            dist2obb  = np.sqrt((tbox[0] + tbox[2] / 2 - obj_bbox[0] - obj_bbox[2] / 2) ** 2
                               +(tbox[1] + tbox[3] / 2 - obj_bbox[1] - obj_bbox[3] / 2) ** 2)
            # Append tracker's tbox for voting new segmented blob
            if psr > self.votePsrThreash and dist2obb < self.voteDistThreash:
                self.votable_tbox_list.append(tbox)
            ave_psr  += psr
            ave_life += tracker.tracked_Frames
            if psr < minpsr:
                minpsr = psr
                minpsr_tracker = tracker  # Prepare to remove the tracker with least psr.
            if psr >= maxpsr:
                maxpsr =  psr
                maxpsr_tracker = tracker  # using for target_rect when there is no qualified voting tracker

        # Fused bounding box for the object using the tracklist.
        obj_bbox = self.fuse_approx_trackers(self.trckList, self.votePsrThreash)
        if obj_bbox is None:
            # if the fusing return none rect, using the tracker with maximum psr as the tail rect in current frame.
            try:
                obj_bbox = maxpsr_tracker.target_rect
            except Exception as e:
                print('obj_box is None')

        # increase the numberofTrackers only when the ave_psr is below 15 (vote_psr is 10)
        ave_psr  = ave_psr  * 1. / len(self.trckList)
        ave_life = ave_life * 1. / len(self.trckList)

        if len(self.trckList) >=  self.dynamicTrckNo:
            if minpsr < self.votePsrThreash:
                self.trckList.remove(minpsr_tracker)
                if DETAIL_MODE == True:
                    print('--Del--minpsr life-%d, psr-%d, size:%s' \
                          % (minpsr_tracker.tracked_Frames, minpsr_tracker.psr,
                             str(minpsr_tracker.target_rect)))

        if ave_psr <= self.votePsrThreash:  # increase the component-tracker's number.
            self.dynamicTrckNo = min(self.dynamicTrckNo + 1, self.maxTrckNo)
            if DETAIL_MODE == True:
                print('Average PSR is below %d, increasing one for maximum tracking numbers, Fuion tracker number is %d'
                % (self.votePsrThreash, self.dynamicTrckNo))

        # Decrease tracker number, minimum is equal to the initial parameter settings.
        if ave_psr > self.votePsrThreash:
            self.dynamicTrckNo = max(self.dynamicTrckNo - 1, self.minTrckNo)
            if DETAIL_MODE == True:
                if self.dynamicTrckNo != self.minTrckNo:
                    print('Decrease one tracker, Fusion tracker number is %d'%self.dynamicTrckNo)

        # Compute the integrated merits  sum(lambda_k) for TAES2021.
        est_x, est_y = [obj_bbox[0] + int(obj_bbox[2] / 2), obj_bbox[1] + int(obj_bbox[3] / 2)]
        lambda_k = 0#1
        res_xs = [] # cord x for all the response matrix' tlx and brx in the big frame
        res_ys = [] # cord y for all the response matrix' tly and bry in the big frame
        # if frame_no==12 and self.tid==91:
        #     #print('')

        # for kcf_trk in self.trckList:
        #     tlx, tly = np.int0([kcf_trk.tlx, kcf_trk.tly])
        #     sh, sw = kcf_trk.window_sz
        #     brx, bry = [tlx+sw, tly+sh]
        #     # tbox = kcf_trk.target_rect
        #     # dist2obb = np.sqrt((tbox[0] + tbox[2] / 2 - obj_bbox[0] - obj_bbox[2] / 2) ** 2
        #     #                    + (tbox[1] + tbox[3] / 2 - obj_bbox[1] - obj_bbox[3] / 2) ** 2)
        #     # # Append tracker's tbox for voting new segmented blob
        #     #if kcf_trk.psr > self.votePsrThreash: #and dist2obb < self.voteDistThreash:
        #     if (tlx <= est_x <= (tlx + sw) and tly <= est_y <= (tly + sh)):
        #         # location of estimated position in component's response matrix
        #         dx, dy = np.int0([est_x - tlx, est_y - tly])
        #         if (dx < sw and dy < sh):# and (kcf_trk.psr> self.votePsrThreash):
        #             # if kcf_trk.response[dy, dx] < 0.5: #location is not reliable in the tracker's response matrix.
        #             #     lambda_k = lambda_k * 0.5#kcf_trk.responsePeak
        #             # else:
        #                 #lambda_k = lambda_k * kcf_trk.response[dy, dx]
        #             #lambda_k += np.log(kcf_trk.response[dy, dx])
        #             lambda_k += np.log(kcf_trk.responsePeak)
        #         else:#no match happens, decrease lambda_k
        #             lambda_k -= 100
        #             print('MKCF_Tid %d Componnet KCF has no lambda_k score.' % self.tid)
        #         res_xs.extend([tlx, brx])
        #         res_ys.extend([tly, bry])
        #     else:
        #         lambda_k -= 100
        #         print('MKCF_Tid %d Componnet KCF is far way to the estimated position' % self.tid)
        #     if self.tid in [0, 1, 2]:
        #         print('MKCF %d lambda_k is %.2f'%(self.tid, np.log(kcf_trk.responsePeak)))
        #lambda_k = np.log(maxpsr_tracker.responsePeak)
        lambda_k  = maxpsr_tracker.responsePeak
        self.merit_list.append(lambda_k)
        self.psr_list.append(maxpsr)

        if len(self.merit_list)<self.integrated_frames:
            self.integrated_merits = np.sum(self.merit_list)
            self.integrated_psrs   = np.sum(self.psr_list)
        else:
            #compute the last K frame's merits
            last_K_merits = np.array(self.merit_list)[-self.integrated_frames:]
            self.integrated_merits = np.sum(last_K_merits)
            self.integrated_psrs   = np.sum(self.psr_list[-self.integrated_frames:])

        # if self.tid in [0, 1, 2]:
        #     #print('MKCF %d maxpsr with PSR %.2f , lambda_k is %.2f'%(self.tid, maxpsr, np.log(maxpsr_tracker.responsePeak)))
        #     print('MKCF %d maxpsr with PSR %.2f , lambda_k is %.2f, mean_lambda: %.2f' % (self.tid, maxpsr, maxpsr_tracker.responsePeak, np.mean(self.merit_list)))

        #print('MKCF_Tid %d, Average PSR %2.2f, merit \u03BB_k %2.4f Average LIFE %3.1f' %
        #      (self.tid, ave_psr, self.integrated_merits, ave_life))

        # if len(res_xs)>0:
        #     res_xs.sort() #sort to find topleft and bottemright
        #     res_ys.sort()
        #     com_res_tlx, com_res_brx = [res_xs[0], res_xs[-1]]
        #     com_res_tly, com_res_bry = [res_ys[0], res_ys[-1]]
        #     combined_response = np.ones((com_res_bry-com_res_tly, com_res_brx-com_res_tlx))
        #     mask = np.zeros((com_res_bry-com_res_tly, com_res_brx-com_res_tlx))
        #     for kcf_trk in self.trckList:
        #         tlx, tly = np.int0([kcf_trk.tlx, kcf_trk.tly])
        #         sh, sw = kcf_trk.window_sz
        #
        #         tbox = kcf_trk.target_rect
        #         dist2obb = np.sqrt((tbox[0] + tbox[2] / 2 - obj_bbox[0] - obj_bbox[2] / 2) ** 2
        #                            + (tbox[1] + tbox[3] / 2 - obj_bbox[1] - obj_bbox[3] / 2) ** 2)
        #         # Append tracker's tbox for voting new segmented blob
        #         if kcf_trk.psr > self.votePsrThreash and dist2obb < self.voteDistThreash:
        #             if (tlx <= est_x <= (tlx + sw) and tly <= est_y <= (tly + sh)): # component rect contains the fusion center
        #                 dx, dy = [tlx - com_res_tlx, tly - com_res_tly]
        #                 assert (dx >= 0 and dy >= 0)
        #                 mask[dy:dy + sh, dx:dx + sw] = kcf_trk.response
        #                 combined_response = combined_response * mask
        #     # Draw the response matrix for interested target
        #     if self.tid == 63:
        #         print('MKCF_Tid %d, Average PSR %2.2f, merit \u03BB_k %2.4f Average LIFE %3.1f' %
        #               (self.tid, ave_psr, self.integrated_merits, ave_life))
        #         kcfs = self.trckList
        #         for i in range(0, min(5, len(kcfs))):
        #             row, col = np.unravel_index(i, (2,3))
        #             mcf_ax[row,col].imshow(kcfs[i].response)
        #             mcf_ax[row,col].title.set_text('psr %.2f'%kcfs[i].psr)
        #         mcf_ax[-1, -1].imshow(combined_response)
        #         mcf_ax[-1, -1].title.set_text('fused response of tid %5d' % self.tid)
        #         mcf_figure.suptitle('frame %d'% frame_no )
        #         plt.draw()
        #         plt.waitforbuttonpress()

        # voted object blob
        obj_blob = {}
        # if len(tracker_list) < NumberOfTrackers:
        # getting the related blob and using the current blob's bounding box to initial the newtracker
        obj_blob, blob_score, blob_id = self.vote_blob(blob_list, self.votable_tbox_list)
        # checking's the selected blob's score, to decide whether init new tracker or not
        # if blob's score is average lower in all trackers, no init.
        # obj_bbox should based on the average of all voted Trackers

        if obj_blob != {}:
            self.voted_blob_id = blob_id
            #blob_bb = obj_blob['BoundingBox']
            blob_bb = obj_blob
            #blob_is_new = True
            # for kcftracker in self.trckList: #check the blob_bb is already contained in the tracker's rect or not
            #     if uti.intersection_area(blob_bb, kcftracker.target_rect)/(blob_bb[2]*blob_bb[3])>0.9:
            #         blob_is_new = False
            if len(self.trckList) < self.dynamicTrckNo:
                blob_tt_iou = uti.intersection_rect(blob_bb, obj_bbox)
                if (DETAIL_MODE == True):
                    print('Tracker %d, blob %04d intersected with obj_bbox %1.2f' % (self.tid, blob_id, blob_tt_iou))
                # only the blob_score enough high, new tracker is initialized
                if (blob_score > self.blobScoreThreash):# and blob_is_new:
                    #newtracker = mosse_tracker.mosse(frame, frame_no, blob_bb)
                    bx,by,bw,bh = blob_bb[:4]
                    bcx = bx + bw/2
                    bcy = by + bh/2
                    bw  = 2*bw
                    bh  = 2*bh
                    #enlarge the initial rectangle with twice width and height.
                    #This method increase the search range of the target
                    #init_rect = [int(blob_bb[0] - blob_bb[2] / 2), int(blob_bb[1] - blob_bb[3] / 2), blob_bb[2]*2, blob_bb[3]*2]
                    init_rect = np.int0([bcx-bw/2, bcy-bh/2, bw, bh])
                    newtracker = kcf_model.KCFTracker(frame, init_rect, frame_no, kernel_opt='gk', kernel_sigma=self.kernel_sigma)
                    self.trckList.append(newtracker)
                    self.init_blob_id = blob_id
                    if (DETAIL_MODE == True):
                        print('Tracker %d, blob %d Adding a new tracker' % (self.tid, blob_id))
                else:
                    if (DETAIL_MODE == True): print('Voted blob is not qualified!')
        else:  # voted blob is null
            self.voted_blob_id = None
            if DETAIL_MODE == True:
                print('No blob is voted!')

        # below self.variable are used for outside operation.
        self.tail_rect_list.append(obj_bbox)
        self.tail_frameno_list.append(frame_no)
        self.ave_psr = ave_psr
        self.ave_life = ave_life


        self.star_peak  = maxpsr_tracker.responsePeak   # the peak of Y^{i^(*)}
        self.star_life  = maxpsr_tracker.tracked_Frames  # the tracked frame numbers for the component tracker with the max_psr.
        self.star_psr   = maxpsr_tracker.psr  # the maximum psr of the component tracker

        self.tracked_Frames += 1
        self.fuse_rect = obj_bbox
        return obj_bbox, ave_psr

    def draw_target_rects(self, frame):
        #  Draw different colors for different PSR for monitoring.
        pass

    def fuse_approx_trackers(self, tracklist, vote_psr_threash=10):
        '''
        Assuming that the trackerlist contain's all the tracker which is greater or equal votePsrThreashold 10,
        then we can approximate the y'_i in Gaussian distribution,
        see the source paper 'MKCF_TSP2019' [Section IV-C] for details.
        :param trackerlist:
        :return: fused bounding box of the obj_bbox.
        '''

        peak_list = []
        tbb_list = []

        for i, tracker in enumerate(self.trckList):
            if tracker.psr >= self.votePsrThreash:
                peak_list.append(tracker.responsePeak)
                tbb_list.append(tracker.target_rect)

        if len(tbb_list) == 0:
            if DETAIL_MODE == True:
                print('Pay attention! No qualified tracker for fusing!!!')
            return None

        peaks = np.array(peak_list)
        tbbs  = np.array(tbb_list)
        weights = peaks ** 2
        weights = weights / np.sum(weights)
        weights = weights.reshape(len(weights), 1)
        weights = np.tile(weights, (1, 4))
        obj_bbox_array = np.int0(np.sum(tbbs * weights, axis=0))
        obj_bbox = [obj_bbox_array[0], obj_bbox_array[1], obj_bbox_array[2], obj_bbox_array[3]]
        assert(obj_bbox[2]*obj_bbox[3]>0)
        if (obj_bbox[0]<0 or obj_bbox[1]<0):
            if DETAIL_MODE == True:
                print('out of range rect ', obj_bbox, self.tid)
            if obj_bbox[0]<0:
                obj_bbox[0] = 0
            if obj_bbox[1]<0:
                obj_bbox[1] = 0
        return obj_bbox

    def vote_blob(self, blob_list, tbox_list):
        '''
        Computing all the bob and tbox's overlapping area, and intersection_ratio,
        Each tracker's tbox vote for a candidate blob.
        The top voted  blob is choosing as the target's blob
        :param blob_list: contains all the blob, each element has a dict('Polygon', 'BoundingBox','Center',.etc), sa blog_geometry
        :param tbox_list: contains all the tracker's estimated bounding box for the target
        :return: (blob_id, blob_score)the blob best matching all the tracker's bbox, measured by overlapped area ratio
        '''

        # Note for empty blob_list or tbox_list
        blen = len(blob_list)
        tlen = len(tbox_list)
        if blen * tlen == 0:
            # return null blob and 0 blob_score
            return {}, 0, None

        scores = np.zeros((tlen, blen), np.float)
        votes = np.zeros((tlen, blen), np.uint)

        # computing each blob and tbox's overlapping ratio, get scores_matrix
        # scores_matrix each row means one tracker in different blob's overlapped ratio
        # each col means one blob in different tracker's bbox's overlapped ratio
        #       scores_matrix(3trackers and 3 blobs)
        #           t\b |  b1 | b2 | b3
        #           t1  | 0.3 |0.2 | 0.1
        #           t2  | 0.1 |0.1 | 0.2
        #           t3  | 0.4 |0.3 | 0.1
        # the blob with the most average score is chosen as the voted blob.
        for i, tbb in enumerate(tbox_list):
            for j, blob in enumerate(blob_list):
                #blob_bb = blob['BoundingBox']
                scores[i, j] = uti.intersection_rect(tbb, blob)

        # vertical suming for counting votes
        # counts = np.sum(votes, axis = 0)

        counts = np.mean(scores, axis=0)
        if np.max(counts)==0: #no overlaps
            # return null blob and 0 blob_score
            return {}, 0, None
        blob_id = np.argmax(counts)
        blob = blob_list[blob_id]
        blob_score = np.mean(scores[:, blob_id])

        # print 'voted blob id %d\n' % blob_id
        # print scores
        return blob, blob_score, blob_id

    def get_target_trajectory(self):
        '''
        Make a trajectory dict by the frame_no key and corresponding rect_list.
        # Output dict's format is the same as the gt_dict in 'simulate_clutter_target_*.py'
        # {'target_name':{frame_no:[rect_x, y, w,h]}}
        :param frameno_list:
        :param rect_list:
        :return:
        '''
        traj_dict = {}
        for frameno, rect in zip(self.tail_frameno_list, self.tail_rect_list):
            # frame_key = '%02d'%frameno
            traj_dict[frameno] = rect
        return traj_dict