# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:25:14 2022

@author: DELL
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
    for i, h_id in enumerate(edge_list[0]): #边列表，形如2*边数
        t_dict[h_id].append(edge_list[1][i]) #尾节点字典：key是头节点ID，value是尾节点ID
        labels_dict[h_id].append(labels[i]) #标签字典：key是头节点ID，value是尾节点与头节点之间有边1或无边0
        conf_dict[h_id].append(confidence[i]) #预测值字典：key是头节点ID,value是尾节点与头节点之间的预测值
    for h_id in t_dict.keys(): #对于每个头节点  
        conf_array = np.array(conf_dict[h_id]) #该头节点与相应尾节点的预测值组成的数组
        
        #---------------更改源码，进行单个正样本与所有负样本比对-------------#
        label_array = np.array(labels_dict[h_id])
        negind = np.where(label_array==0) #找到所有负样本
        neg_array = conf_array[negind] #负样本得分
        pos_index = []
        pos_index_c = []
        for y_ in range(label_array.shape[0]):
            if label_array[y_] != 0:
                neg_array = np.hstack((conf_array[y_],neg_array)) #一个正样本+所有负样本
                rank = np.argsort(-neg_array)
                pos_index0 = np.where(rank == 0)[0][0]
                pos_index0_c = 1/(1+pos_index0)
                pos_index.append(pos_index0)
                pos_index_c.append(pos_index0_c)
        if len(pos_index) == 0: #如果为空，表明全是负样本，继续
            continue
        pos_sum_rank = np.mean(pos_index)
        cur_mrr = np.mean(pos_index_c)
        #---------------更改源码，进行单个正样本与所有负样本比对-------------#
        
        
        # rank = np.argsort(-conf_array) #进行排序，本来是升序排列，加上负号，变成降序排列
        # sorted_label_array = np.array(labels_dict[h_id])[rank] #该头节点对应的标签字典中标签排序
        # pos_index = np.where(sorted_label_array == 1)[0] #查找降序排列的标签中值为1的位置索引
        # if len(pos_index) == 0: #如果为空，表明全是负样本，继续
        #     continue
        # # pos_min_rank = np.min(pos_index) #返回最小的索引值，即该头节点对应的第一个正样本的预测值越靠前越好
        # pos_sum_rank = np.sum(pos_index)
        # cur_mrr = 1/(1+pos_sum_rank)
        mrr_list.append(cur_mrr)
        mr_list.append(pos_sum_rank+1)
    mrr = np.mean(mrr_list)
    mr = np.mean(mr_list)
    return  mrr, mr
    #     cur_mrr = 1 / (1 + pos_min_rank)
    #     mrr_list.append(cur_mrr)
    # mrr = np.mean(mrr_list)
    # return {'roc_auc': roc_auc,'roc_pr': auc_pr, 'MRR': mrr}
    
def WKNKN(Y, SD, ST, K, eta):
    Yd = np.zeros(Y.shape)
    Yt = np.zeros(Y.shape)
    wi = np.zeros((K,))
    wj = np.zeros((K,))
    num_drugs, num_targets = Y.shape
    for i in np.arange(num_drugs):
        dnn_i = np.argsort(SD[i,:])[::-1][1:K+1]
        Zd = np.sum(SD[i, dnn_i])
        for ii in np.arange(K):
            wi[ii] = (eta ** (ii)) * SD[i,dnn_i[ii]]
        if not np.isclose(Zd, 0.):
            Yd[i,:] = np.sum(np.multiply(wi.reshape((K,1)), Y[dnn_i,:]), axis=0) / Zd
    for j in np.arange(num_targets):
        tnn_j = np.argsort(ST[j, :])[::-1][1:K+1]
        Zt = np.sum(ST[j, tnn_j])
        for jj in np.arange(K):
            wj[jj] = (eta ** (jj)) * ST[j,tnn_j[jj]]
        if not np.isclose(Zt, 0.):
            Yt[:,j] = np.sum(np.multiply(wj.reshape((1,K)), Y[:,tnn_j]), axis=1) / Zt
    Ydt = (Yd + Yt)/2
    x, y = np.where(Ydt > Y)

    Y_tem = Y.copy()
    Y_tem[x, y] = Ydt[x, y]
    return Y_tem


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
            # ma[yd[-nei:],i]=1
        elif sum(ma1[i]>0)>5:
            yd=np.argsort(ma1[i])
            ma[i,yd[-5:]]=1
            # ma[yd[-nei:],i]=1
        elif sum(gip[i]>0)>2:
            yd=np.argsort(gip[i])
            ma[i,yd[-2:]]=1
            # ma[yd[-nei:],i]=1
    return ma

def Cosin(A):
    mm=A.shape[0]
    KM=np.zeros((mm,mm))
    for cd in range(mm):
        for dc in range(0,cd+1):
            if (LA.norm(A[cd,:],ord=2)*LA.norm(A[dc,:],ord=2)==0):
                KM[cd,dc] =0
            else:
                multi=np.dot(A[cd,:],A[dc,:].T)
                KM[cd,dc]= multi/(LA.norm(A[cd,:],ord=2)*LA.norm(A[dc,:],ord=2))
            KM[dc,cd]= KM[cd,dc]
    dd=A.shape[1]
    KD=np.zeros((dd,dd))
    for ab in range(dd):
        for ba in range(0,ab+1):
            if (LA.norm(A[:,ab],ord=2)*LA.norm(A[:,ba],ord=2)==0):
                KD[ab,ba] =0
            else:
                dmulti=np.dot(A[:,ab].T,A[:,ba])
                KD[ab,ba]= dmulti/(LA.norm(A[:,ab],ord=2)*LA.norm(A[:,ba],ord=2))
            KD[ba,ab]= KD[ab,ba]
    return np.array(KM),np.array(KD)




def random_walk_with_restart(interaction):
    p = 0.9  #0.9
    iter_max = 1000
    origi_matrix = np.identity(interaction.shape[0])
    sum_col = interaction.sum(axis=0)  
    sum_col[sum_col == 0.] = 2
    interaction = np.divide(interaction,sum_col)
    pre_t = origi_matrix
    
    for i in range(iter_max):
#        print("i:",i)
        t = (1-p) * (np.dot(interaction, pre_t)) + p * origi_matrix
        pre_t = t
    return t

def extract_global_neighbors(interaction, walk_matrix):
    interaction = interaction.astype(int)
    interaction_mask = np.zeros_like(interaction)
    neigh_index = np.argsort(-walk_matrix, axis=0) #按照每列升序排序返回索引值   
    
    for j in range(interaction.shape[1]):      
        for i in range(np.sum(interaction[j,:])):   #选择top m个作为全局近邻，论文使用直接邻居数量作为m值
            interaction_mask[neigh_index[i,j],j]=1
    return interaction_mask.T
   
            
def RWR(M):#邻接矩阵经过随机游走后产生的全局关联矩阵
    walk_matrix = random_walk_with_restart(M)
    interaction_global = extract_global_neighbors(M, walk_matrix)
    return interaction_global



def Crr(A):
    noise = np.random.normal(loc=0,scale=1.0,size=A.shape)
    A = A +noise
    score1 = np.corrcoef(A)
    score2 = np.corrcoef(A.T)    
    return score1,score2
    
    