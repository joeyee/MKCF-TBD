'''
This File implement the Dynamic programming algorithm for Track-Before-Detect in TSP2013 of Grossi.
Exhaustive search the trajectory which is physically constrained,
in the aim of finding the maximum test statistics.
Illustrate the Track formation and pruning via tracking
simulated multiple extended-targets in clutter background.
Author: Yi Zhou,
Creation date: 2020-12-29
'''

import numpy as np
from   matplotlib.patches import Rectangle
import utilities_200611          as uti            # personal tools
import matplotlib.pyplot         as plt
import taes2021_utility_20210216 as sp_model       # support tools for TAES2021

import cv2
# Python model and tools written by myself.
#import mosse_tracker_20200601     as mosse_model    # Single correlated tracker
#import MCF_20200603               as mcf_model      # Fuse multiple correlated trackers for a target
import cfar_segmentation_200527   as cfar_model     # cfar
#import svm_learning_200610        as svm_model      # customized svm
#import vessel_detect_track_200614 as vdt_model      # jdat's ancestor
import evaluate_results_200623    as eval_model     # evaluate the tracking results.
import motion_simulation_20201030 as msim_model     # Motion simulation of multiple targets.
import utilities_200611           as uti            # personal tools
import time
import csv
from matplotlib.patches import Rectangle


class Nodes():
    def __init__(self):
        self.pre_list = [] # previous nodes's name list
        self.nxt_list = [] # next nodes list
        self.tra_list = [] # Available Trajectory list of current node.
        self.tau = []      # best trajectory which is with the maximum accumulated values.
        self.val = 0       # value of node, average power of the average amplitude in a plot.
        self.fkl = 0       # accumulated test statistics F_kl
        self.name= ''      # name string, 'bxxx' (blob-blodid%03d) for observed target, 'zxxx' (zero for alarm caused by noise) for background alarm node.
        self.fid = 0       # frame no of this node
        self.pos = (0,0)   # position (x,y) coordinates or (r, theta)
        self.wh  = (0,0)   # width and height of a plot
        self.am  = 0       # average echo intensity amplitude in a plot
        self.noise= 1      # background noise power.
        #self.plotstate = {} # plot state dict {'t': ,'r':, 'theta':, 'am':, 'noise':} of Eq.(1) in paper add width and height

class DP_TBD_Grossi():
    '''
    Implementation of the Grossi's Dynamic programming in paper Grossi_TSP2013
    '''
    def __init__(self, P=4, Q=3, L=10, gamma2 = 4):

        self.P = P #maximum number of consecutive misses in Track formation
        self.Q = Q #minimum number of blob observatoins in Track pruning for each trajectory( make sure the acceleration could be checked)
        self.L = L #integrated frames for TBD.
        self.gamma2 = gamma2
        self.vmax = 30 #maximum speed
        self.amax = 20 #maximum acceleration
        self.nodes_dict={} # store all the node(include the plot state) information of L frames in dict
                           # {'fid%03d':[nodeslist], 'fid':[nodeslist, ..., 'fid':[nodeslist]]}
        self.target_trajectory_dict = {} #two dimensional dict. {'tid':traj_dict{'fid':tbox}}
        self.target_tau_dict = {}        #{'tid':tau}
        self.tidnums  = 0  # confirmed trajectory associated with tid.
        self.dist_thresh = 15*2   #distance threshold for measuring two neighbour nodes.
                                  #import setting for avoid mismatch in ship-meeting
        self.fidlist = [] #store all the frame id in list

    def nodes_distance(self, nd_apl, nd_bab):
        '''
        Measure the distance of two nodes  according to the position.
        :param nd_apl:
        :param nd_bab:
        :return:
        '''
        cor_apl = np.array(nd_apl.pos)
        cor_bab = np.array(nd_bab.pos)
        dist    = np.sqrt(np.sum((cor_apl - cor_bab)**2))
        return dist

    def nodes_velocity_distance(self, nd_apl, nd_bab):
        pass

    def nodes_accelerate_distance(self, nd_apl, nd_bab):
        pass

    def remove_least_node_link(self, node, nodes_dict):
        '''
        Remove a node from a nodes_dict's link.
        set the connected node's previous node list = null.
        remove the node from last frame's node's tau list.
        :param node:
        :param nodes_dict:
        :return:
        '''
        nd_fid   = node.fid
        last_fid = max(nodes_dict.keys())
        assert(nd_fid < last_fid) # Not removing the last frame's node.
        connect_nodes_list = nodes_dict[nd_fid+1]

        #remove from the previous link list.
        for con_node in connect_nodes_list:
            if node in con_node.pre_list:
                con_node.pre_list.remove(node)

        #remove from the last_node's best trajectory list.
        last_nodes_list = nodes_dict[last_fid]
        for lastfid_node  in last_nodes_list:
            if node in lastfid_node.tau:
                lastfid_node.tau.remove(node)

    def remove_least_frame_nodes(self, nodes_dict):
        '''
        remove the least's frame's nodes, clear their connections to latter nodes.
        clear the last frame's nodes best trajectory tau.
        :param nodes_dict:
        :return:
        '''
        fids = nodes_dict.keys()
        least_fid = min(fids)
        last_fid  = max(fids)

        for least_fid_node in nodes_dict[least_fid]:
            self.remove_least_node_link(least_fid_node, nodes_dict)

        del nodes_dict[least_fid]

    def get_ave_clutter_power(self, frame, bloblist):
        '''
        compute the average clutter power in each frame.
        :param frame:
        :param bloblist:
        :return:
        '''
        clutter_mask =  np.ones_like(frame)
        for blob_rect in bloblist:
            x,y,w,h = blob_rect
            clutter_mask[y:y+h, x:x+w] = 0 #omit the target's rectangle region.
        ave_bk_power = np.sum((frame*clutter_mask)**2)/np.sum(clutter_mask)
        self.sigma_n = ave_bk_power
        return ave_bk_power

    def generate_nodes(self, frame, fid, bloblist, timestep=1):
        '''
        This function is used to generate nodes in each frame, given the segmented bloblist.
        store it in self.nodes_dict[fid]
        :param fid: frame No.
        :param bloblist: segmented blob(tlx, tly, w, h) for each plot.
        :param timestep: time step between two frames
        :return: nodes list in current frame.
        '''
        bid = 0
        nodes_list = []
        self.fidlist.append(fid)
        #compute the background clutter's average power and upate sigma_n
        self.sigma_n = self.get_ave_clutter_power(frame, bloblist)
        # if fid > self.L: # store more than L frame, need to delete first frame
        #     self.remove_least_frame_nodes(self.nodes_dict)
        for blobrect in bloblist:
            tlx, tly, w, h = blobrect
            if w*h ==0 :
                continue # omit the point target.
            bid += 1
            blobroi = frame[tly:tly+h, tlx:tlx+w]
            #Target location is at the maximum response
            #row, col = np.unravel_index(blobroi.argmax(), blobroi.shape)
            # plot_row = tly + row
            # plot_col = tlx + col
            plot_time = fid*timestep
            plot_r    = tly+int(h/2)
            plot_theta= tlx+int(w/2)
            new_node = Nodes()
            new_node.name = 'B%03d'%bid #f:fid,b:bid
            new_node.fid  = fid
            new_node.pos  = (plot_r, plot_theta)
            new_node.wh   = (w,h)
            new_node.am   = np.average(blobroi)
            new_node.val  = np.average(blobroi**2)#new_node.am**2
            # print('%s mean square amplitude %.5f, snr=%.2f'
            #      %(new_node.name, new_node.val, 10*np.log10(new_node.val/2)))
            #if fid == 0:  # first frame start from 0 in simulated targets, but from 1 in inesa
            if len(self.fidlist)==1:
                new_node.fkl = new_node.val    # init the accumulated test statistics F_{kl}
                #new_node.tau.append(new_node)  # add self as the first elements in the first frame.
            nodes_list.append(new_node)
        self.nodes_dict[fid] = nodes_list
        return nodes_list

    def find_neighbour(self, fid, nodes_dict):
        '''
        Obey physical constraints, get neighbours for each node (pre or next nodes), in current layer.
        :return:
        '''
        dist_thresh = self.dist_thresh
        #assert(layerid>1)

        #if fid > 0: # find neighour starts from the second layer.
        if len(self.fidlist)>1:
            nodes_pre = nodes_dict[fid-1]
            nodes_cur = nodes_dict[fid]
            # first round loop, set the linked previous and next for two layers
            for nd_pre  in nodes_pre:
                for nd_cur in nodes_cur:
                    if self.nodes_distance(nd_pre, nd_cur)<= dist_thresh:
                        # find the next nodes for each nd in nodes_pre.
                        # at the same time set the pre nodes for cur node
                        nd_pre.nxt_list.append(nd_cur)
                        nd_cur.pre_list.append(nd_pre)

            # next round loop, set the nodes of previous layer with no linked next nodes to zero-nodes.
            zeros_num = 0 # number of zero_nodes.
            for nd_pre in nodes_pre:
                if len(nd_pre.nxt_list)==0: # no linked next nodes
                    zeros_num += 1
                    zero_node = Nodes() # generate new zero node in current layer.
                    zero_node.pos = nd_pre.pos  # equal to the previous node's position
                    zero_node.val = zero_node.noise  # equal to noise's intensity.
                    zero_node.fkl = nd_pre.fkl + zero_node.val
                    zero_node.name= 'Z%03d'%zeros_num # zero node, false alarm caused by noisy clutter:zero_id
                    zero_node.fid = fid
                    nd_pre.nxt_list.append(zero_node)
                    zero_node.pre_list.append(nd_pre)
                    nodes_dict[fid].append(zero_node)

    def generate_trajectory(self, fid, nodes_dict):
        '''
        Generate all the trajectories from the nodes of last frame to those in current frame.
        Compute the accumulated test statistics F_{kl} and best trajectory
        for each node in current layer.
        :return:
        '''
        nodes_cur = nodes_dict[fid]
        for nd_cur in nodes_cur:
            if len(nd_cur.pre_list) == 0:
                # node without previous link, is a new begin trajectory  for its successors.
                nd_cur.fkl = nd_cur.val
                nd_cur.tau.append(nd_cur)
            else:
                nd_cur.tra_list = []  # clear all the trajectory list of layerid's node, store all the compatible trajectories.
                Fkl = 0  # maximum the test statistic F_kl.
                maxVal = 0
                maxFkl = 0
                maxid  = None
                for id, nd_pre in enumerate(nd_cur.pre_list):
                    # Set the trajectory list of each current node
                    # Need to check the nd_pre in nd
                    # Note current traj list is based on previous best trajectory: tau.
                    tra = nd_pre.tau.copy()  # Note! only best trajectory tau is copied.
                    # tra.append(nd_cur)          # add current node
                    nd_cur.tra_list.append(tra)  # add to current trajectory list.
                    if(nd_pre.val<0):
                        print(nd_pre)
                    if maxVal <= nd_pre.val:
                        maxVal = nd_pre.val
                        maxid  = id
                    # if maxFkl <= nd_pre.fkl:
                    #     maxFkl = nd_pre.fkl
                    #     maxid  = id
                # Set the test statistics F_kl based on the max value of the linked previous node.
                assert(maxid is not None)
                best_pre_node = nd_cur.pre_list[maxid]
                nd_cur.fkl    = best_pre_node.fkl + nd_cur.val
                # chose the maximum F_kl traj as the best trajectory.
                nd_cur.tau    = nd_cur.tra_list[maxid]
                nd_cur.tau.append(nd_cur)

    def is_zero_node(self, node):
        '''
        Judege the node is zero_node or not. zero node is generated by the previous node with no linked next.
        the zero_node's name is 'Z(zero_num_id)layerid' begins with letter 'Z'.
        :param node:
        :return:
        '''
        name = node.name
        # zero_node's name's first element is not a alphabet.
        if name[0] == 'Z':
            return True
        else:
            return False

    def get_max_fkl_node(self, nodes_list):
        '''
        return the node with maximum fkl in a list
        :param nodes_list:
        :return:
        '''
        maxfkl = 0
        maxnode = None
        for nd in nodes_list:
            if maxfkl <= nd.fkl:
                maxfkl = nd.fkl
                maxnode = nd
        return maxnode

    def count_zero_nodes(self, nodes_list):
        '''
        count how many zero nodes in a list
        :param nodes_list:
        :return:
        '''
        n = 0
        for nd in nodes_list:
            if self.is_zero_node(nd):
                n += 1
        return n

    def count_consecutive_zero_nodes(self, nodes_list):
        '''
        count consecutive zero nodes start from the first non-zero node in  a trajectory node list
        :param nodes_list:
        :return:
        '''
        cz = 0
        bstart = False
        max_cz = 0            #confirmed trajectory should have max_cz <= self.P
        for nd in nodes_list:
            if self.is_zero_node(nd) == False: #first non-zero nodes
                bstart = True
                cz = 0
                continue
            if self.is_zero_node(nd)  and bstart:
                cz += 1
                if max_cz < cz:
                    max_cz = cz
                continue
            if self.is_zero_node(nd) == False and bstart: # stop count the zero nodes.
                bstart = False
        return max_cz

    def add_zero_nodes(self, nodes_dict, fid):
        '''
        add a zero node in a layerid of nodes_dict
        :param nodes_dict:
        :param layerid:
        :return:
        '''
        zero_node = Nodes()  # generate new zero node in current layer.
        zero_node.val = 0
        zero_node.fkl = 0
        zero_node.name = 'Z%03d' % (self.count_zero_nodes(nodes_dict[fid]) + 1)
        return nodes_dict

    def prun_trajectory(self, nodes_dict):
        '''
        Prun the trajectory in L-th Frame, where two trajectory shares a common root.
        This is a implementation of Algorithm 2 in GrossiTsp2013 paper.
        Prun happens in two conditions:
        1.Two trajectories share same roots, add zero_nodes(false alarm is the source)
        to the trajectory with less fkl.
        2.trajectory has less non-zero nodes than Q(Q=3 for layers=10). low quality trajectory.
        :param nodes_dict:
        :return:
        '''
        prun_nodes_dict = self.nodes_dict #nodes_dict.copy(), not copy.
        keys = prun_nodes_dict.keys()
        lastfid = max(keys)

        nodes_lastframe = nodes_dict[lastfid]
        nodes_prunlist = nodes_lastframe.copy()
        while len(nodes_prunlist) > 0:
            maxnode = self.get_max_fkl_node(nodes_prunlist)
            #print('zero nodes count for maxnode %d'%self.count_zero_nodes(maxnode.tau))
            nodes_prunlist.remove(maxnode)
            if len(maxnode.tau) <= self.L:  #omit the maxnode with less than L trajectory nodes.
                continue
            for nd in nodes_prunlist:
                if len(nd.tau)  <= self.L:  # omit the node with less than L trajectory nodes.
                    continue
                l = 0
                bprun = False # prune the nd node's trajectory or not?
                try:
                    while nd.tau[l] == maxnode.tau[l]:  # nd share the same roots with maxnode
                        bprun = True # prune the nd's trajectory
                        # two end-node in frame L share same root in frame l,
                        # keep the maxnode's tau, prun other node's tau, fill with 0.
                        assert (len(nd.tau) == len(maxnode.tau))
                        fid = int(nd.tau[l].fid)
                        zero_node = Nodes()  # generate new zero node in current layer.
                        zero_node.pos = nd.tau[l].pos  # equal to the next node's position
                        zero_node.val = zero_node.noise
                        zero_node.fkl = nd.tau[l].fkl - nd.tau[l].val + zero_node.val # refresh fkl for the adding zero-node
                        zero_node.fid = fid
                        zero_node.name= 'Z%03d' % (self.count_zero_nodes(prun_nodes_dict[fid]) + 1)
                        nd.tau[l]     = zero_node
                        prun_nodes_dict[fid].append(zero_node)
                        l = l + 1
                        # if l > lastlayer_id:
                        #     print('%s and %s has the same tau trajectory!!!'%(nd.name, maxnode.name))
                except Exception as e:
                    print(e)

                if bprun:
                    traj_id = 0
                    # connect the newly added zero nodes to the next node. omit the last node.
                    # the link is connected via the modified nd.tau trajectory list.
                    while traj_id < (len(nd.tau) - 1):
                        nd_traj_cur = nd.tau[traj_id]
                        nd_traj_nxt = nd.tau[traj_id + 1]
                        if len(nd_traj_cur.nxt_list) == 0:  # newly added zero node
                            nd_traj_cur.nxt_list.append(nd_traj_nxt)
                            nd_traj_nxt.pre_list.append(nd_traj_cur)
                        traj_id += 1

                # prun the trajectory with less than Q non-zero nodes
                zero_nodes_in_traj = self.count_zero_nodes(nd.tau)
                non_zero_nums = len(nd.tau) - zero_nodes_in_traj
                #Q = 3 in default L=10 frames.
                if non_zero_nums < self.Q:
                    nd.tau[-2].nxt_list.remove(nd) #remove current node from pre_node.nx_list
                    nd.tau      = [nd]  # prun the trajectory with only current node.
                    nd.pre_list = []    # cut the previous list of nd
                    bprun       = True
                if bprun:
                    # recompute the fkl in the pruned node.
                    fkl = 0
                    for nd_traj in nd.tau:
                        fkl += nd_traj.val  # recompute fkl in the pruned trajectory.
                    nd.fkl = fkl
        self.nodes_dict = prun_nodes_dict
        return prun_nodes_dict

    def confirm_trajectory(self, lastfid, nodes_dict, gamma2=4, bshow_fkl=False):
        '''
        confirm trajectory based on the accumulated fkl (test statistics)
        :param nodes_dict:
        :return:
        '''
        lastfid_nodes_list = nodes_dict[lastfid]
        for nd in lastfid_nodes_list:
            if len(nd.tau) >= self.L: # integration should be long enough
                #confirmed trajectory satisfy three conditions : fkl is big, consecutive zeros is low
                # and non-zeros is high
                fkl = 0
                for i in range(-1,-(self.L+1),-1):
                    fkl += nd.tau[i].val
                ave_power = fkl/self.L
                nd.fkl = fkl

                if(bshow_fkl):
                # uncomment following line to estimate the gamma_threshold.
                    print('Candidate %s traj\'s length %d - FKL %.2f snr %.2f'
                           %(nd.name, len(nd.tau),nd.fkl, 10*np.log10(ave_power/self.sigma_n)))
                #background power is 2*Rayleigh_scale^2
                if nd.fkl > gamma2 and self.count_consecutive_zero_nodes(nd.tau) <= self.P\
                        and self.count_zero_nodes(nd.tau) <= (len(nd.tau) - self.Q):
                    # Following print monitoring the snr changes of the tracked target's roi
                    # print('Confirmed %s traj\'s length %d in frame %2d - FKL %.2f snr %.2f'
                    #       % (nd.name, len(nd.tau), nd.fid, nd.fkl, 10 * np.log10(ave_power / 2)))
                    bexist = False # judge nd.tau's origin already in target_tau_dict or not
                    exist_times = 0
                    for key in self.target_tau_dict:
                        confirm_tau = self.target_tau_dict[key]
                        if confirm_tau[-1] == nd.tau[-2]:
                            # confirmed tau's last node equal to current node's tau's previous node.
                            bexist = True
                            self.target_tau_dict[key] = nd.tau #update target's tau. not change the tid
                            exist_times+=1 #only exist once in confirm_tau
                            if (bshow_fkl):
                                print('Tid %d traj\'s length %d - FKL %.2f snr %.2f'
                                      % (key, len(nd.tau), nd.fkl, 10 * np.log10(ave_power / self.sigma_n)))
                    assert(exist_times<2)
                    if not bexist:
                        self.tidnums += 1  # add new tid
                        self.target_tau_dict[self.tidnums] = nd.tau #give traj to new tid.


    def transfer_tau_to_trajectory_dict(self, target_tau_dict):
        #Transfer confirmed node's tau into trajectory_dict{'fid':tbox}
        #Store the trajectory into target_traj_dict{'tid':trajectory_dict}

        for tid in target_tau_dict:
            #key is tid
            tau = target_tau_dict[tid]
            traj_dict = {}
            for traj_nd in tau:
                fid    = traj_nd.fid
                cy, cx = traj_nd.pos
                w,   h = traj_nd.wh
                tbox= [cx-int(w/2), cy-int(h/2), w, h]
                #traj_key =  '%02d'%fid #compatible with the evaluation model, change the int key to string
                traj_key = fid
                traj_dict[traj_key] = tbox
            self.target_trajectory_dict[tid] = traj_dict

    def print_nodes_dict(self, nodes_dict):
        '''
        print the dictionary information
        :param nodes_dict:
        :return:
        '''
        for key in nodes_dict:
            print('layer %s' % key)
            nodes_list = nodes_dict[key]
            for node in nodes_list:
                print('%s fkl: %3.1f' % (node.name, node.fkl))
                traj_names = ''
                if len(node.tau) > 0:
                    for nd in node.tau:
                        # if nd==0: # zero node
                        #     traj_names +=' 0 '
                        # else:
                        traj_names += (' %s ' % nd.name)
                else:
                    traj_names = 'None'
                print(traj_names)

    def draw_nodes_dict(self, nodes_dict):
        '''
        Draw the connection graph of the nodes_dict.
        :param nodes_dict:
        :return:
        '''
        fig, ax = plt.subplots()
        keys    = nodes_dict.keys()

        margin = 50
        width  = (len(keys) + 2) * margin

        maxNodesInColuomn = 0
        for key in nodes_dict:
            columnHeight = len(nodes_dict[key])
            if maxNodesInColuomn < columnHeight:
                maxNodesInColuomn = columnHeight
        height = (maxNodesInColuomn + 2) * margin

        for layerid in nodes_dict:
            nodes_layer = nodes_dict[layerid]
            for i, node in enumerate(nodes_layer):
                plt.plot(layerid * margin, (i + 1) * margin, 'o')
                plt.text(layerid * margin - 10, (i + 1) * margin - 20,
                         '%s[%3.1f]\n%3.1f' % (node.name, node.val, node.fkl), fontsize=8)
                for nd_nx in node.nxt_list:
                    nx_column = nodes_dict[layerid + 1].index(nd_nx)
                    plt.plot([layerid * margin, (layerid + 1) * margin], [(i + 1) * margin, (nx_column + 1) * margin],
                             '-', alpha=0.3)

        # draw the best trajectory in the last layer.
        lastlayer = max(keys)
        for node_lastlayer in nodes_dict[lastlayer]:
            x = []
            y = []
            for traj_node in node_lastlayer.tau:
                layerid = int(traj_node.fid)
                try:
                    column = nodes_dict[layerid].index(traj_node)
                except Exception as e:
                    print(e)
                x.append(layerid * margin)
                y.append((column + 1) * margin)
            plt.plot(x, y, '-.', linewidth=3)

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # plt.show()
    def draw_traj(self, ax, fid, traj_dict, offset_xy = (0,0)):
        '''
        Draw traj on current fid's frame
        :param self:
        :param fid:
        :return:
        '''
        offset_x, offset_y = offset_xy # offset of the topleft x and y.
        for tid in traj_dict:
            tau = traj_dict[tid]
            color_tuple = np.random.random(3).tolist() #one target using one color
            for id,nd in enumerate(tau):
                y, x = nd.pos  # center position
                x    = x - offset_x
                y    = y - offset_y
                w, h = nd.wh
                if id==1: # First node write down the tid in the origin
                    #tbox = [x - int(w / 2), y - int(h / 2), w, h]
                    ax.text(x, y, '%d'%tid, color=(1, 1, 1, 1), fontsize=6)
                if nd.fid == fid: #draw current rectangle on this fid
                    rect_patch = Rectangle(xy=[x-w/2, y-h/2], width=w, height=h, angle=0, lw=2, color=color_tuple,fill=None)
                    ax.add_patch(rect_patch)
                #draw pos point
                rect_patch = Rectangle(xy=[x-1, y-1], width=2, height=2, angle=0, color=color_tuple,fill=None)
                ax.add_patch(rect_patch)


def draw_bounding_boxs(frame, bloblist, ax, color=(255,255,0), offset_xy = (0,0)):
    '''
    Draw bounding boxs of the bloblist given the frame on canvas in cv2.
    :param frame:
    :param bloblist:
    :return:
    '''
    # draw in cv2.
    # uframe = uti.frame_normalize(frame) * 255
    # uframe = uframe.astype(np.uint8)
    # canvas = cv2.applyColorMap(uframe, cv2.COLORMAP_JET)
    # for blob in bloblist:
    #     uti.draw_rect(canvas, blob, color=color)  # draw classified blob in light blue
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # fig1, ax1 = plt.subplots()
    # ax1.imshow(canvas)

    # draw in plt
    # fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(frame)
    offset_x, offset_y = offset_xy
    for blob in bloblist:
        tlx, tly, w, h = blob
        rect = Rectangle(xy=[tlx-offset_x, tly-offset_y], width=w, height=h, color=color, fill=None)
        ax.add_artist(rect)
    #plt.show()


def test_dp_tbd_parameter_verbose(integrated_frames, snr=10, gamma2=200, swerling_type=0, kval=1.8):
    '''
    Check parameters of the dynamic tbd by tracking the simulated multiple extended targets.
    # snr means signal to noise ratio in the simulated clutter background.
    :return:
    '''
    # 1st generate the targets and gt_information
    #gt_dict = msim_model.multiple_extended_targets_in_clutter()
    gt_rotate_rect_trajectory_dict = msim_model.multiple_extended_targets_in_clutter()
    gt_trajecotry_dict = sp_model.convert_rotateRect_to_Rect_trajectory_dict(gt_rotate_rect_trajectory_dict)
    #gt_trajecotry_dict = gt_rotate_rect_trajectory_dict
    frame_nums = len(gt_rotate_rect_trajectory_dict.keys())
    img_w = 300
    img_h = 300
    bVerbose = False  #print detail information for diagnose or not

    msim_model.local_snrs = [] #clear the  local snrs records
    msim_model.global_snrs= [] #clear the  msim_model's global snr records.

    #snr   = 0  # signal to noise ratio in the simulated clutter background.
    # Illustrate all the simulated frames.
    # fig, ax = plt.subplots()
    # for frame_no in range(nframes):
    #     print('frame %2d - %d'%(frame_no, nframes))
    #     frame = msim_model.get_frame(img_w, img_h, frame_no, snr, gt_dict)
    #     plt.imshow(frame)
    #     plt.pause(0.1)

    ## Parameters for Vessel Detect Tracker (VDT)
    # Qualified continuously tracking frames.
    # cfar_seg_kval = 1.8  # 1.8  # 1.8      # 1.2  for inesa, 1.3 for taes20_sim
    # cfar_seg_max_area = 600  # 200+ for inesa, 100  for taes20_sim
    # cfar_ref = 20  # best setting is 20
    # cfar_guide = 10  # best setting is 12
    #
    #GrossiTSP2013 Parameters setting for cfar
    # cfar_seg_kval                     = 1.8 #1.8
    # cfar_seg_max_area                 = 600   # 200+ for inesa, 100  for taes20_sim
    # cfar_seg_min_area                 = 32     # 200+ for titan, 100  for TriFalo
    # cfar_ref                          = 20    # Titan 20， TriFalo 10.
    # cfar_guide                        = 10      # Titan 10,  TriFalo 8.

    # #MCF Parameter setting for cfar
    cfar_seg_kval                     = kval#1.3    # 1.0  for Titan, 1.3 for TriFalo
    cfar_seg_max_area                 = 30000   # 200+ for inesa, 100  for taes20_sim
    cfar_seg_min_area                 = 90     # 200+ for titan, 100  for TriFalo
    cfar_ref                          = 16*2     # Titan 20， TriFalo 10.
    cfar_guide                        = 8*2      # Titan 10,  TriFalo 8.

    dp_tbd = DP_TBD_Grossi(P=4, Q=3, L=integrated_frames, gamma2=gamma2)
    fig, ax = plt.subplots()
    fid_cfar_list     = []
    pfa_cfar_list     = []
    pd_cfar_list      = []
    pfa_seg_list      = []
    pd_seg_list       = []
    time_counter_list = []  # contains the time consum for each frame.

    for fid in range(frame_nums):

        frame = msim_model.get_frame(img_w, img_h, fid, snr, gt_rotate_rect_trajectory_dict, swerling_type)  # gray_scale image
        # uframe = uti.frame_normalize(frame) * 255
        # uframe = uframe.astype(np.uint8)
        # # canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # canvas = cv2.applyColorMap(uframe, cv2.COLORMAP_JET)
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)     # for plt.imshow()
        blob_bb_list, bin_image = cfar_model.segmentation(frame, lbp_contrast_select=False,
                                                          kval=cfar_seg_kval, least_wh=(3, 3), nref=cfar_ref,
                                                          mguide=cfar_guide,
                                                          min_area=cfar_seg_min_area, max_area=cfar_seg_max_area)
        print('Frame %02d cfar segments %4d blobs' % (fid, len(blob_bb_list)))

        key = '%02d' % fid
        assert (key in gt_rotate_rect_trajectory_dict)
        targets_regions_dict = {}
        # # get each frame's multiple target's vertex for computing the pfa_pd of cfar and clustering.
        # for tname in gt_dict[key]:
        #     x, y, w, h = gt_dict[key][tname][:4]
        #     targets_regions_dict[tname] = [[x, y], [x + w, y], [x, y + h],
        #                                    [x + w, y + h]]  # using 4 vertex of rectangles
        pfa_cfar, pd_cfar, pfa_seg, pd_seg = \
            sp_model.get_pfa_pd_via_cfar_rrect(bin_image, gt_rotate_rect_trajectory_dict['%02d' % fid], blob_bb_list)
        fid_cfar_list.append(fid)
        pfa_cfar_list.append(pfa_cfar)
        pd_cfar_list.append(pd_cfar)
        pfa_seg_list.append(pfa_seg)
        pd_seg_list.append(pd_seg)

        tcost_frame = time.perf_counter()
        dp_tbd.generate_nodes(frame, fid, blob_bb_list, timestep=1)
        dp_tbd.find_neighbour(fid, dp_tbd.nodes_dict)
        dp_tbd.generate_trajectory(fid, dp_tbd.nodes_dict)
        if fid >= (dp_tbd.L - 1):  # fid start from 0
            # trajectory formation in last frame's node.tau list.
            # dp_tbd.generate_trajectory(fid, dp_tbd.nodes_dict)
            # prune trajectory and confirm the trajectory
            # dp_tbd.print_nodes_dict(dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(dp_tbd.nodes_dict)
            prun_nodes_dict = dp_tbd.prun_trajectory(dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(prun_nodes_dict)
            dp_tbd.confirm_trajectory(fid, dp_tbd.nodes_dict, dp_tbd.gamma2)
        tcost_frame = time.perf_counter() - tcost_frame
        time_counter_list.append(tcost_frame)

        print('Frame %02d cfar segments %4d blobs, consume %.2f seconds' % (fid, len(blob_bb_list), tcost_frame))
        print('Frame %02d pfa_cfar %.5f, pfa_seg %.5f, pd_cfar %.5f, pd_seg %.5f' % (fid, pfa_cfar, pfa_seg, pd_cfar, pd_seg))
        # draw segmented bounding box in light blue via cv2.
        draw_bounding_boxs(frame, blob_bb_list, ax, color=(0, 1, 1))
        dp_tbd.draw_traj(ax, fid, dp_tbd.target_tau_dict)
        fig.canvas.set_window_title('Frame %d'%fid)
        plt.pause(0.01)

    dp_tbd.transfer_tau_to_trajectory_dict(dp_tbd.target_tau_dict)

    #targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(gt_dict)
    targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(gt_trajecotry_dict)
    precision_dict, false_alarm_acc, overall_roh_dict = eval_model.measure_trajectory_precesion(targets_gt_dict,
                                                                                                dp_tbd.target_trajectory_dict)

    eval_model.print_metrics(precision_dict, false_alarm_acc, img_w, img_h, frame_nums, overall_roh_dict)
    eval_model.draw_track_traj(img_w, img_h, targets_gt_dict, dp_tbd.target_trajectory_dict, precision_dict)

    # dp_tbd.draw_nodes_dict(dp_tbd.nodes_dict)

    # compute pfa and pd via track_trajectory and gt_trajectory
    frame_track_trajectory = sp_model.convert_target_trajectory_to_frame_trajectory(dp_tbd.target_trajectory_dict,
                                                                                    bshrink=False)
    fid_trk_list, pfa_trk_list, pd_trk_list = sp_model.get_pfa_pd_via_trajectory_rrect(frame_track_trajectory,
                                                                                 gt_trajecotry_dict,
                                                                                 frame_height = img_h, frame_width=img_w,
                                                                                bshrink_tbox=False)
    print('CFAR AVE Pfa_cfar %.5f, Cfar AVE Pd %.5f' % (np.mean(pfa_cfar_list), np.mean(pd_cfar_list)))
    print('Seg  AVE Pfa_Seg  %.5f, Seg  AVE Pd %.5f' % (np.mean(pfa_seg_list),  np.mean(pd_seg_list )))
    print('Trck AVE Pfa_trck %.5f, Trk  AVE Pd %.5f' % (np.mean(pfa_trk_list), np.mean(pd_trk_list)))
    print('False Alarm Reduction Gain %.2fdb' % (10 * np.log10(np.mean(pfa_seg_list) / np.mean(pfa_trk_list))))
    print('Detection Increase Gain    %.2fdb' % (10 * np.log10(np.mean(pd_trk_list) / np.mean(pd_seg_list))))
    print('Average consuming time is %.2f per frame, total consuming time is %.2f' % (
        np.mean(time_counter_list), np.sum(time_counter_list)))
    fig_pfa, pfaax = plt.subplots()
    fig_pd, pdax = plt.subplots()
    pfaax.plot(fid_cfar_list, pfa_cfar_list, color=(0, 0, 0.5), label='pfa_cfar[%.5f]' % np.mean(pfa_cfar_list))
    pfaax.plot(fid_cfar_list, pfa_seg_list, color=(0, 0.5, 0), label='pfa_seg[%.5f]' % np.mean(pfa_seg_list))
    pfaax.plot(fid_trk_list, pfa_trk_list, color=(1, 0, 0), label='pfa_trk [%.5f]' % np.mean(pfa_trk_list))
    pfaax.legend()
    pdax.plot(fid_cfar_list, pd_cfar_list, color=(0, 0, 0.5), label='pd_cfar[%2.2f]' % np.mean(pd_cfar_list))
    pdax.plot(fid_cfar_list, pd_seg_list, color=(0, 0.5, 0), label='pd_seg[%2.2f]' % np.mean(pd_seg_list))
    pdax.plot(fid_trk_list, pd_trk_list, color=(1, 0, 0), label='pd_trk[%2.2f]' % np.mean(pd_trk_list))
    pdax.legend()
    plt.pause(0.01)
    plt.waitforbuttonpress()
    # plt.show()

def test_dp_tbd_parameter(integrated_frames, snr=10, gamma2=100, swerling_type=0):
    '''
    Check parameters of the dynamic tbd by tracking the simulated multiple extended targets.
    # snr means signal to noise ratio in the simulated clutter background.
    :return:
    '''
    print('integrated frames:%d, gamma =%.2f, snr = %d, swerling_type %d'%(integrated_frames,gamma2, snr, swerling_type))
    # 1st generate the targets and gt_information
    #gt_dict = msim_model.multiple_extended_targets_in_clutter()
    gt_rotate_rect_trajectory_dict = msim_model.multiple_extended_targets_in_clutter()
    gt_trajecotry_dict = sp_model.convert_rotateRect_to_Rect_trajectory_dict(gt_rotate_rect_trajectory_dict)
    #gt_trajecotry_dict = gt_rotate_rect_trajectory_dict
    frame_nums = len(gt_rotate_rect_trajectory_dict.keys())
    img_w = 300
    img_h = 300
    bVerbose = False  #print detail information for diagnose or not

    msim_model.local_snrs = [] #clear the  local snrs records
    msim_model.global_snrs= [] #clear the  msim_model's global snr records.

    #snr   = 0  # signal to noise ratio in the simulated clutter background.
    # Illustrate all the simulated frames.
    # fig, ax = plt.subplots()
    # for frame_no in range(nframes):
    #     print('frame %2d - %d'%(frame_no, nframes))
    #     frame = msim_model.get_frame(img_w, img_h, frame_no, snr, gt_dict)
    #     plt.imshow(frame)
    #     plt.pause(0.1)

    ## Parameters for Vessel Detect Tracker (VDT)
    # Qualified continuously tracking frames.
    # cfar_seg_kval = 1.8  # 1.8  # 1.8      # 1.2  for inesa, 1.3 for taes20_sim
    # cfar_seg_max_area = 600  # 200+ for inesa, 100  for taes20_sim
    # cfar_ref = 20  # best setting is 20
    # cfar_guide = 10  # best setting is 12
    #
    #GrossiTSP2013 Parameters setting for cfar
    # cfar_seg_kval                     = 1.8 #1.8
    # cfar_seg_max_area                 = 600   # 200+ for inesa, 100  for taes20_sim
    # cfar_seg_min_area                 = 32     # 200+ for titan, 100  for TriFalo
    # cfar_ref                          = 20    # Titan 20， TriFalo 10.
    # cfar_guide                        = 10      # Titan 10,  TriFalo 8.

    # #MCF Parameter setting for cfar
    cfar_seg_kval                     = 1.3#1.3    # 1.0  for Titan, 1.3 for TriFalo
    cfar_seg_max_area                 = 30000   # 200+ for inesa, 100  for taes20_sim
    cfar_seg_min_area                 = 90     # 200+ for titan, 100  for TriFalo
    cfar_ref                          = 16*2     # Titan 20， TriFalo 10.
    cfar_guide                        = 8*2      # Titan 10,  TriFalo 8.

    dp_tbd = DP_TBD_Grossi(P=4, Q=3, L=integrated_frames, gamma2=gamma2)
    #fig, ax = plt.subplots()
    fid_cfar_list     = []
    pfa_cfar_list     = []
    pd_cfar_list      = []
    pfa_seg_list      = []
    pd_seg_list       = []
    time_counter_list = []  # contains the time consum for each frame.

    for fid in range(frame_nums):

        frame = msim_model.get_frame(img_w, img_h, fid, snr, gt_rotate_rect_trajectory_dict, swerling_type)  # gray_scale image
        # uframe = uti.frame_normalize(frame) * 255
        # uframe = uframe.astype(np.uint8)
        # # canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # canvas = cv2.applyColorMap(uframe, cv2.COLORMAP_JET)
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)     # for plt.imshow()
        blob_bb_list, bin_image = cfar_model.segmentation(frame, lbp_contrast_select=False,
                                                          kval=cfar_seg_kval, least_wh=(3, 3), nref=cfar_ref,
                                                          mguide=cfar_guide,
                                                          min_area=cfar_seg_min_area, max_area=cfar_seg_max_area)
        # print('Frame %02d cfar segments %4d blobs' % (fid, len(blob_bb_list)))

        key = '%02d' % fid
        assert (key in gt_rotate_rect_trajectory_dict)
        targets_regions_dict = {}
        # # get each frame's multiple target's vertex for computing the pfa_pd of cfar and clustering.
        # for tname in gt_dict[key]:
        #     x, y, w, h = gt_dict[key][tname][:4]
        #     targets_regions_dict[tname] = [[x, y], [x + w, y], [x, y + h],
        #                                    [x + w, y + h]]  # using 4 vertex of rectangles
        pfa_cfar, pd_cfar, pfa_seg, pd_seg = \
            sp_model.get_pfa_pd_via_cfar_rrect(bin_image, gt_rotate_rect_trajectory_dict['%02d' % fid], blob_bb_list)
        fid_cfar_list.append(fid)
        pfa_cfar_list.append(pfa_cfar)
        pd_cfar_list.append(pd_cfar)
        pfa_seg_list.append(pfa_seg)
        pd_seg_list.append(pd_seg)

        tcost_frame = time.perf_counter()
        dp_tbd.generate_nodes(frame, fid, blob_bb_list, timestep=1)
        dp_tbd.find_neighbour(fid, dp_tbd.nodes_dict)
        dp_tbd.generate_trajectory(fid, dp_tbd.nodes_dict)
        if fid >= (dp_tbd.L - 1):  # fid start from 0
            # trajectory formation in last frame's node.tau list.
            # dp_tbd.generate_trajectory(fid, dp_tbd.nodes_dict)
            # prune trajectory and confirm the trajectory
            # dp_tbd.print_nodes_dict(dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(dp_tbd.nodes_dict)
            prun_nodes_dict = dp_tbd.prun_trajectory(dp_tbd.nodes_dict)
            # dp_tbd.draw_nodes_dict(prun_nodes_dict)
            dp_tbd.confirm_trajectory(fid, dp_tbd.nodes_dict, dp_tbd.gamma2)
        tcost_frame = time.perf_counter() - tcost_frame
        time_counter_list.append(tcost_frame)

        # print('Frame %02d cfar segments %4d blobs, consume %.2f seconds' % (fid, len(blob_bb_list), tcost_frame))
        # print('Frame %02d pfa_cfar %.5f, pfa_seg %.5f, pd_cfar %.5f, pd_seg %.5f' % (fid, pfa_cfar, pfa_seg, pd_cfar, pd_seg))
        # # draw segmented bounding box in light blue via cv2.
        # draw_bounding_boxs(frame, blob_bb_list, ax, color=(0, 1, 1))
        # dp_tbd.draw_traj(ax, fid, dp_tbd.target_tau_dict)
        # fig.canvas.set_window_title('Frame %d'%fid)
        # plt.pause(0.01)

    dp_tbd.transfer_tau_to_trajectory_dict(dp_tbd.target_tau_dict)

    #targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(gt_dict)
    targets_gt_dict = eval_model.reform_multiple_extended_targets_gt(gt_trajecotry_dict)
    precision_dict, false_alarm_acc, overall_roh_dict = eval_model.measure_trajectory_precesion(targets_gt_dict,
                                                                                                dp_tbd.target_trajectory_dict)

    _, _, res_table = eval_model.print_metrics(precision_dict, false_alarm_acc, img_w, img_h, frame_nums, overall_roh_dict)
    #eval_model.draw_track_traj(img_w, img_h, targets_gt_dict, dp_tbd.target_trajectory_dict, precision_dict)

    # dp_tbd.draw_nodes_dict(dp_tbd.nodes_dict)

    # compute pfa and pd via track_trajectory and gt_trajectory
    frame_track_trajectory = sp_model.convert_target_trajectory_to_frame_trajectory(dp_tbd.target_trajectory_dict,
                                                                                    bshrink=False)
    fid_trk_list, pfa_trk_list, pd_trk_list = sp_model.get_pfa_pd_via_trajectory_rrect(frame_track_trajectory,
                                                                                 gt_trajecotry_dict,
                                                                                 frame_height = img_h, frame_width=img_w, bshrink_tbox=False)
    print('CFAR AVE Pfa_cfar %.5f, Cfar AVE Pd %.5f' % (np.mean(pfa_cfar_list), np.mean(pd_cfar_list)))
    print('Seg  AVE Pfa_Seg  %.5f, Seg  AVE Pd %.5f' % (np.mean(pfa_seg_list),  np.mean(pd_seg_list )))
    print('Trck AVE Pfa_trck %.5f, Trk  AVE Pd %.5f' % (np.mean(pfa_trk_list), np.mean(pd_trk_list)))
    print('False Alarm Reduction Gain %.2fdb' % (10 * np.log10(np.mean(pfa_seg_list) / np.mean(pfa_trk_list))))
    print('Detection Increase Gain    %.2fdb' % (10 * np.log10(np.mean(pd_trk_list) / np.mean(pd_seg_list))))
    print('Average consuming time is %.2f per frame, total consuming time is %.2f' % (
        np.mean(time_counter_list), np.sum(time_counter_list)))
    # fig_pfa, pfaax = plt.subplots()
    # fig_pd, pdax = plt.subplots()
    # pfaax.plot(fid_cfar_list, pfa_cfar_list, color=(0, 0, 0.5), label='pfa_cfar[%.5f]' % np.mean(pfa_cfar_list))
    # pfaax.plot(fid_cfar_list, pfa_seg_list, color=(0, 0.5, 0), label='pfa_seg[%.5f]' % np.mean(pfa_seg_list))
    # pfaax.plot(fid_trk_list, pfa_trk_list, color=(1, 0, 0), label='pfa_trk [%.5f]' % np.mean(pfa_trk_list))
    # pfaax.legend()
    # pdax.plot(fid_cfar_list, pd_cfar_list, color=(0, 0, 0.5), label='pd_cfar[%2.2f]' % np.mean(pd_cfar_list))
    # pdax.plot(fid_cfar_list, pd_seg_list, color=(0, 0.5, 0), label='pd_seg[%2.2f]' % np.mean(pd_seg_list))
    # pdax.plot(fid_trk_list, pd_trk_list, color=(1, 0, 0), label='pd_trk[%2.2f]' % np.mean(pd_trk_list))
    # pdax.legend()
    # plt.pause(0.01)
    # plt.waitforbuttonpress()
    # plt.show()
    pfa_gain = 10 * np.log10(np.mean(pfa_seg_list) / np.mean(pfa_trk_list))
    pd_gain  = 10 * np.log10(np.mean(pd_trk_list) / np.mean(pd_seg_list))

    record = [swerling_type, snr, np.mean(time_counter_list), np.mean(pfa_cfar_list), np.mean(pfa_seg_list),
              np.mean(pfa_trk_list), pfa_gain,
              np.mean(pd_cfar_list), np.mean(pd_seg_list), np.mean(pd_trk_list), pd_gain,
              integrated_frames, gamma2, cfar_seg_kval, cfar_ref, cfar_guide,
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
    return record_txt

if __name__ == '__main__':
    test_dp_tbd_parameter_verbose(integrated_frames=9, snr=10, gamma2=22, swerling_type=0, kval=1.3)
    # field names
    fields = ['SwerlingType', 'SNR', 'TimeCost', 'Pfa_cfar', 'Pfa_seg', 'Pfa_trk','Pfa_gain',
              'pd_cfar', 'pd_seg', 'pd_trk', 'pd_gain',
              'IntegratedFrames', 'Gamma', 'cfar_kval', 'ref_cells', 'guide_cells',
              'ave_epos', 'ave_iou', 'ave_roh', 'ave_ntf',
              'vic_epos', 'vic_iou', 'vic_roh', 'vic_ntf',
              'ame_epos', 'ame_iou', 'ame_roh', 'ame_ntf',
              'uri_epos', 'uri_iou', 'uri_roh', 'uri_ntf', ]
    # name of csv file
    filename = "/Users/yizhou/code/taes2021/results/DP_TBD_GROSSI_SIMULATE.csv"
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        snrs = list(range(12, -3, -1))  # [12, 11, ..., -1, -2]
        for sw_type in [0, 1, 3]:
            for snr in snrs:
                record = test_dp_tbd_parameter(integrated_frames=9, gamma2=26, snr=snr, swerling_type=sw_type)
                # writing the data rows
                csvwriter.writerow(record)
                csvfile.flush()
    print('Parameter_test complete! Remember to modify the name of csv for saving records!')