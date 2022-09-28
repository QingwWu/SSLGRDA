# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:25:14 2022

@author: wu
"""

import numpy as np
from collections import defaultdict
import numpy.linalg as LA


def evalu(edge_list, confidence, labels):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param labels: shape(edge_num,)
    :return: dict with all scores we need
    """
    confidence = np.array(confidence.flatten())
    labels = np.array(labels.flatten())
    mrr_list, mr_list,cur_mrr = [],[], 0
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]): 
        t_dict[h_id].append(edge_list[1][i]) 
        labels_dict[h_id].append(labels[i]) 
        conf_dict[h_id].append(confidence[i]) 
    for h_id in t_dict.keys():   
        conf_array = np.array(conf_dict[h_id]) 

        label_array = np.array(labels_dict[h_id])
        negind = np.where(label_array==0) 
        neg_array = conf_array[negind] 
        pos_index = []
        pos_index_c = []
        for y_ in range(label_array.shape[0]):
            if label_array[y_] != 0:
                neg_array = np.hstack((conf_array[y_],neg_array)) 
                rank = np.argsort(-neg_array)
                pos_index0 = np.where(rank == 0)[0][0]
                pos_index0_c = 1/(1+pos_index0)
                pos_index.append(pos_index0)
                pos_index_c.append(pos_index0_c)
        if len(pos_index) == 0: 
            continue
        pos_sum_rank = np.mean(pos_index)
        cur_mrr = np.mean(pos_index_c)
        mrr_list.append(cur_mrr)
        mr_list.append(pos_sum_rank+1)
    mrr = np.mean(mrr_list)
    mr = np.mean(mr_list)
    return  mrr, mr

def GIP(A):#A is numpy 2D array
    gamad1=1
    sumk1=0
    ss=A.shape[1]
    for nm in range(ss):
        sumk1=sumk1+LA.norm(A[:,nm],ord=2)**2
    gamaD1=gamad1*ss/sumk1
    KD=np.mat(np.zeros((ss,ss)))
    for ab in range(ss):
        for ba in range(ss):
            KD[ab,ba]=np.exp(-gamaD1*LA.norm(A[:,ab]-A[:,ba])**2)
    gamad2=1
    sumk2=0
    mm=A.shape[0]
    for mn in range(mm):
        sumk2=sumk2+LA.norm(A[mn,:],ord=2)**2
    gamaD2=gamad2*mm/sumk2
    KM=np.zeros((mm,mm))
    for cd in range(mm):
        for dc in range(mm):
            KM[cd,dc]=np.exp(-gamaD2*LA.norm(A[cd,:]-A[dc,:])**2)
    return np.array(KM),np.array(KD)

def toplink(ma1,gip,nei=5):
    for i in range(ma1.shape[0]):
        ma1[i,i]=0
        gip[i,i]=0 
    ma=np.zeros((ma1.shape[0],ma1.shape[1]))
    for i in range(ma1.shape[0]):
        if sum(ma1[i]>0)>nei:
            yd=np.argsort(ma1[i])
            ma[i,yd[-nei:]]=1
        elif sum(ma1[i]>0)>5:
            yd=np.argsort(ma1[i])
            ma[i,yd[-5:]]=1
    return ma
    
    
