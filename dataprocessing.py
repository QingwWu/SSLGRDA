# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:51:43 2022

@author: USTC
"""



import pickle
import numpy as np
from tools import GIP,toplink
import math
import scipy.sparse as sp
import random


def dataload(data,num):
    #--------------------------------- miRNA-disease------------------------------------#
    if data =='mda':
        if num =='1':
            ld=np.loadtxt('D:/bioinformatics/1_data/MDA/mda.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/MDA/sd.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/MDA/sm.csv', delimiter=',') 
        elif num=='2':
            sd=np.loadtxt('D:/bioinformatics/1_data/MDA/51_kernels/sd_fun.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/MDA/51_kernels/sm_fun.csv', delimiter=',')     
            m=sl.shape[0]
            d=sd.shape[0]
            ld=np.zeros((m,d))
            labels = np.loadtxt("D:/bioinformatics/1_data/MDA/51_kernels/adj.txt")
            B=labels.astype(int)
            for b in B:
                ld[b[0]-1,b[1]-1]=1      
        elif num =='3':
            ld=np.loadtxt('D:/bioinformatics/1_data/MDA/HMDD v3.2/md3.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/MDA/HMDD v3.2/sd3.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/MDA/HMDD v3.2/sm3.csv', delimiter=',') 
     #-------------------------------------- circRNA-disease-----------------------------#    
    elif data =='cda':
        if num =='1':     
            ld=np.loadtxt('D:/bioinformatics/1_data/CDA/51_kernels/cda.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/CDA/51_kernels/sd.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/CDA/51_kernels/sc.csv', delimiter=',')
        elif num=='2':
            ld=np.loadtxt('D:/bioinformatics/1_data/CDA/CircR2Disease/cda.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/CDA/CircR2Disease/sd.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/CDA/CircR2Disease/sc.csv', delimiter=',') 
        elif num =='3':
            ld=np.loadtxt('D:/bioinformatics/1_data/CDA/GMNN/cd.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/CDA/GMNN/sc.csv', delimiter=',') 
            sd=np.loadtxt('D:/bioinformatics/1_data/CDA/GMNN/sd.csv', delimiter=',')             
    #----------------------------- lncRNA-disease------------------------------------#  
    elif data =='lda':
        if num =='1':     
            ld=np.loadtxt('D:/bioinformatics/1_data/LMDA/newlda2.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/LMDA/newsd2.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/LMDA/sl.csv', delimiter=',')
        elif num=='2':
            ld=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/ld_89_190.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/sd_190.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/sl_89.csv', delimiter=',')
        elif num=='3':
            ld=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/ld_194_128.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/sl_194.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/LMDA/sdata/sd_128.csv', delimiter=',') 
    #==================-----------==== microbe-drug =========-------==================#
    elif data =='memr':
        if num =='1': 
            ld=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/MDAD/MD.csv', delimiter=',')
            sd=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/MDAD/sd.csv', delimiter=',')
            sl=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/MDAD/sm.csv', delimiter=',')  
        elif num =='2':
            sd=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/aBiofilm/drugsimilarity.txt')
            sl=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/aBiofilm/microbesimilarity.txt')
            lda = np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/aBiofilm/adj.txt')
            ld = np.zeros((sl.shape[0],sd.shape[0]))
            for b in lda:
                ld[int(b[1])-1,int(b[0])-1]=1    
        elif num =='3':
            sd=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/DrugVirus/drugsimilarity.txt')
            sl=np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/DrugVirus/microbesimilarity.txt')
            lda = np.loadtxt('D:/bioinformatics/1_data/MeDrA/33_data/DrugVirus/adj.txt')
            ld = np.zeros((sl.shape[0],sd.shape[0]))
            for b in lda:
                ld[int(b[1])-1,int(b[0])-1]=1
    #================================= microbe-disease ===========================#
    elif data =='medi':
        if num =='1':     
            sd = np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/Disbiome/disease_features.txt")
            sl= np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/Disbiome/microbe_features.txt")
            m=sl.shape[0]
            d=sd.shape[0]
            ld=np.zeros((m,d))
            labels = np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/Disbiome/adj.txt")
            B=labels.astype(int)
            for b in B:
                ld[b[1]-1,b[0]-1]=1     
        elif num =='2':
            sd = np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/HMDAD/disease_features.txt")
            sl= np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/HMDAD/microbe_features.txt")
            m=sl.shape[0]
            d=sd.shape[0]
            ld=np.zeros((m,d))
            labels = np.loadtxt("D:/bioinformatics/1_data/mircorbe-dis/HMDAD/adj.txt")
            B=labels.astype(int)
            for b in B:
                ld[b[1]-1,b[0]-1]=1    

    ##### 保存已知关联，方便主代码导入 ###########    
    spall = sp.csr_matrix(ld)    
    file3 = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/allMat.pkl','wb')
    pickle.dump(spall,file3)
    file3.close()      
   
    file4 = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/GH/Data/rda/allMat.pkl','wb')
    pickle.dump(spall,file4)
    file4.close()   


    #######  分割测试集，并置已知关联矩阵中对应值为0 ############    
    a,b = np.where(ld==1)
    assnum = a.size
    tesnum = math.floor(assnum*0.2)
    lis=[]
    while(len(lis)<tesnum):
        ind = np.random.randint(0,assnum)
        if ind not in lis:
            lis.append(ind)
    a = a[lis]
    b = b[lis]   
    tmat = np.zeros_like(ld)
    for i in range(a.size):
        ld[a[i],b[i]] =0
        tmat[a[i],b[i]] =1
        
    #-------------- 使用其他划分方式，每个节点至少保留一个边，没有孤立点 ------------------#
    # col = np.sum(ld,axis=0)
    # row = np.sum(ld,axis=1)
    # ass = int(np.sum(ld)*0.2)
    # ridx,cidx = np.where(ld==1)
    # nridx,ncidx = np.where(ld==0)
    
    # testedge =[]
    # pait=0
    # tmat = np.zeros_like(ld)
    # while(len(testedge)<ass):
    #     idx = np.random.randint(0,ridx.size)
    #     ri = ridx[idx]
    #     ci = cidx[idx]
    #     if row[ri]>1 and col[ci]>1:
    #         testedge.append([ri,ci])
    #         tmat[ri,ci] =1
    #         ld[ri,ci]=0
    #         row[ri] = row[ri]-1
    #         col[ci] = col[ci]-1
    #         ridx = np.delete(ridx,idx)
    #         cidx = np.delete(cidx,idx)
    #     else:
    #         pait+=1
    #     if pait>10000:
    #         print('超过迭代冗余次数%d,未能生成已知样本的1/5'%pait)
    #         break   
    #-------------- 使用其他划分方式，每个节点至少保留一个边，没有孤立点 ------------------#
    
    ######## 生成GIP并填充相似矩阵，然后生成topN边，并构建同构网络
    gipl,gipd = GIP(ld)
    sl1 = toplink(sl,gipl,5)
    sd1 = toplink(sd,gipd,5)
    h1 = np.hstack((sl1,ld))
    h2 = np.hstack((ld.T,sd1))
    h = np.vstack((h1,h2))
    tstmat = h[0:ld.shape[0],ld.shape[0]:]
    ######生成训练边列表 并保存txt    
    row,col = np.where(h==1)
    edge=[]
    for i in range(row.size):
        edge.append((row[i],col[i]))
    np.savetxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge.txt',edge,fmt="%d")
    
    row0,col0 = np.where(ld==1)
    edge0=[]
    for i in range(row0.size):
        edge0.append((row0[i],col0[i]+ld.shape[0]))
        # edge0.append((row0[i],col0[i]+ld.shape[0]))
        # edge0.append((row0[i],col0[i]+ld.shape[0]))
        # edge0.append((row0[i],col0[i]+ld.shape[0]))
    np.savetxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge0.txt',edge0,fmt="%d") 
 
    row1,col1 = np.where(ld==0)
    edge1=[]
    indlist = []
    for i in range(row0.size):
        ind  = np.random.randint(0,row1.size)
        if ind not in indlist:
            edge1.append((row1[ind],col1[ind]+ld.shape[0]))
    np.savetxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge1.txt',edge1,fmt="%d")     
    
    idxlist =np.zeros((h.shape[0],5))
    for i in range(h.shape[0]):
        ro = np.where(h[i]==0)[0]
        ro1 = random.sample(list(ro),5)
        idxlist[i] = ro1
    np.savetxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/idxlist.txt',idxlist,fmt="%d") 
        
    ####### 保存测试集矩阵，方便导入 #########
    sptmat = sp.csr_matrix(tmat)
    filet2 = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/tMat.pkl','wb')
    pickle.dump(sptmat,filet2)
    filet2.close()
    filet5 = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/GH/Data/rda/tMat.pkl','wb')
    pickle.dump(sptmat,filet5)
    filet5.close()    
    

    zemat = np.zeros_like(ld)
    fea1 = np.hstack((sl,zemat))
    fea2 = np.hstack((zemat.T,sd))
    fea = np.vstack((fea1,fea2))  
    np.savetxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/fea.txt',fea)   
    
    #------------- GH data save-------------------#    
    sph = sp.csr_matrix(h)
    fileh = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/GH/Data/rda/trnMat.pkl','wb')
    pickle.dump(sph,fileh)
    fileh.close()
    
    sptest = sp.csr_matrix(tstmat)
    filet1 = open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/GH/Data/rda/tstMat.pkl','wb')
    pickle.dump(sptest,filet1)
    filet1.close()   
    
    
    
    
    print('同构图数据已经处理好')
    
    
    

