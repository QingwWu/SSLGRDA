# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:19:35 2022

@author: Wu
"""
import argparse
from ranking import mean_rank,mean_reciprocal_ranks,average_precision
import pickle
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,average_precision_score,roc_curve,auc,precision_recall_curve
from sklearn.metrics import recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
import numpy as np
import pandas as pd
from tools import evalu
import random
import tensorflow as tf

from GM.SSLG_GM_homo import GM_homo
from GM.SSLG_GM_hete import GM_hete
from GM.SSLG_GM_homo import GM_get_args

from GH.SSLG_GH_homo import GH_homo
from GH.SSLG_GH_hete import GH_hete
from GH.DataHandler import DataHandler
from dataprocessing import dataload
from dataprocessing_hete import dataload_hete

from MA.utils import build_args
from MA.SSLG_MA_homo import MA_homo
from MA.SSLG_MA_hete import MA_hete
from MA.SSLG_MA_hete import MA_get_args

parser = argparse.ArgumentParser(description='Parser for SSLGRDA')
parser.add_argument('--mt', type=str, default='MA', help='model type',choices=['GM','GH','MA'])
parser.add_argument('--gt', type=str, default='hete', help='graph type',choices=['homo','hete'])
parser.add_argument('--ds', type=str, default='lda', help='datasets',choices=['cda','lda','mda','medr','medi'])
parser.add_argument('--dn', type=str, default='2', help='data number',choices=['1','2','3'])
args = parser.parse_args()


allauc=0
allpr=0
allmr =0
allmrr = 0
lmrr=0
lmr=0
hist10 = 0
hist50 = 0
hist100 = 0
f1score = 0
f1score2 = 0
lmrr2=0
lmr2=0
prec = 0
reca = 0
auclist,prlist = [],[]
for loop in range(5):
    if args.mt =='GM':
        GM_args = GM_get_args()
        if args.gt == 'homo':
            fea,_ = GM_homo(GM_args, args.ds,args.dn, gpu_id=None)
        elif args.gt == 'hete':
            fea,_ = GM_hete(GM_args, args.ds,args.dn,gpu_id=None)  
    elif args.mt =='GH':
        if args.gt == 'homo':
            dataload(args.ds,args.dn)
            handler = DataHandler()
            handler.LoadData()
            tf.reset_default_graph()
            with tf.Session() as sess:
                recom = GH_homo(sess, handler)
                allpredresult,allfea,gallfea,hallfea = recom.run()
                fea = 0*hallfea[1]+0*hallfea[0]+gallfea[1]+gallfea[0]
        elif args.gt == 'hete':
            dataload_hete(args.ds,args.dn)
            handler = DataHandler()
            handler.LoadData()
            tf.reset_default_graph()
            with tf.Session() as sess:
                recom = GH_hete(sess, handler)
                allpredresult,allfea,gallfea,hallfea = recom.run()            
                fea = np.vstack((gallfea[0],gallfea[1]))
    elif args.mt =='MA':
        if args.gt == 'homo':
            MAargs = build_args()
            fea = MA_homo(MAargs,args.ds,args.dn).numpy()
        elif args.gt == 'hete':
            MA_args = MA_get_args()
            fea = MA_hete(MA_args,args.ds,args.dn,gpu_id=None)
            
    with open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/tMat.pkl', 'rb') as fs:
    	testMat = pickle.load(fs).todense()
    with open('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/allMat.pkl', 'rb') as fs:
    	rd = pickle.load(fs).todense()    
    m=rd.shape[0]
    d=rd.shape[1] 
   
    rfea = fea[0:m]
    dfea = fea[m:]
    fullknown = []
    testknown = []
    for ind in range(0,m*d):
        j=ind%d
        i=int(ind/d)
        if (rd[i,j]==1):
            fullknown.append(ind)
        if (testMat[i,j]==1):
            testknown.append(ind)
    allInd=list(range(0,m*d))
    allInd1=list(set(allInd).difference(set(fullknown)))
    #随机生成和已知关联索引同样大小的负样本索引
    knownInd = list(set(fullknown).difference(set(testknown)))
    negInd=random.sample(allInd1,len(knownInd))
    # negativeSampleIndices=random.sample(allIndices1,int(0.2*len(allIndices1)))
    #所有样本索引（含已知关联和负样本索引）
    posNegInd=knownInd+negInd
    #所有未知索引
    allnegs= allInd1#list(set(allIndices1).difference(set(negativeSampleIndices)))
    posNegRNA=[]
    posNegDis=[]             
    for inde in range(0,len(posNegInd)):
        l=posNegInd[inde]%d
        k=int(posNegInd[inde]/d)
        posNegRNA.append(k)
        posNegDis.append(l)
    
    ########################特征向量####################################
    RNAFeature = rfea[posNegRNA]
    DisFeature = dfea[posNegDis]        
    trainFeature = np.hstack((RNAFeature,DisFeature))

    
    knowlabel=np.ones((len(knownInd),1))
    neglabel=np.zeros((len(negInd),1))
    y_train=np.concatenate((knowlabel,neglabel),axis=0)
   
    clf = ExtraTreesClassifier(n_estimators=1000)
    clf.fit(trainFeature, y_train.ravel())
    
    
    tposNegRNA=[]
    tposNegDis=[]
    for inde in range(0,len(testknown)):
        l=testknown[inde]%d
        k=int(testknown[inde]/d)
        tposNegRNA.append(k)
        tposNegDis.append(l)
    tRNAfea = rfea[tposNegRNA]
    tDisfea = dfea[tposNegDis]
    tFeature=np.hstack((tRNAfea,tDisfea))
    tledge = np.vstack((np.array(tposNegRNA),np.array(tposNegDis)+m))
    tledge2 = np.vstack(((np.array(tposNegDis)+m),np.array(tposNegRNA)))
    
    gposNegRNA=[]
    gposNegDis=[]
    for inde in range(0,len(allnegs)):
        l=allnegs[inde]%d
        k=int(allnegs[inde]/d)
        gposNegRNA.append(k)
        gposNegDis.append(l)
    gRNAfea = rfea[gposNegRNA]
    gDisfea = dfea[gposNegDis]
    # m5cv=np.random.randint(gRNAfea.shape[0],size=1*tFeature.shape[0])
    m5cv = np.random.randint(gRNAfea.shape[0],size=int(0.2*gRNAfea.shape[0]))
    #-------------上面的随机生成会有重复值-----------------#
    # m5cv0 = list(np.arange(gRNAfea.shape[0]))
    # random.shuffle(m5cv0)
    # m5cv = m5cv0[:int(0.2*gRNAfea.shape[0])]
    
    gFeature=np.hstack((gRNAfea,gDisfea))[m5cv]
    tgedge = np.vstack((np.array(gposNegRNA),np.array(gposNegDis)+m)).T[m5cv].T
    tgedge2 = np.vstack(((np.array(gposNegDis)+m),np.array(gposNegRNA))).T[m5cv].T
    tedge = np.hstack((tledge,tgedge))
    tedge2 = np.hstack((tledge2,tgedge2))      

    
    X_test=np.vstack((tFeature,gFeature))
    test_P_label=np.ones((tFeature.shape[0],1))
    test_N_label=np.zeros((gFeature.shape[0],1))
    y_test=np.concatenate((test_P_label,test_N_label),axis=0)

    ###################  嵌入表示可视化  ##############################  
    # x_tnse = TSNE(n_components=2,random_state=33).fit_transform(trainFeature)
    # inix_tnse = TSNE(n_components=2,random_state=44).fit_transform( X_test)
    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.scatter(x_tnse[:,0],x_tnse[:,1],c=y_train.ravel(), label='t-SNE')
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(inix_tnse[:,0],inix_tnse[:,1],c=y_test.ravel(), label='test-SNE')
    # plt.legend()        
    # plt.show()  
      
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc_roc = roc_auc_score(y_test.ravel(), y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    ps, rs, _ = precision_recall_curve(y_test, y_pred_proba)
    aupr = auc(rs, ps)
    
    pscore = precision_score(y_test, y_pred)
    rscore = recall_score(y_test, y_pred)
    prec +=pscore
    reca +=rscore
    print(auc_roc, auc_pr,pscore,rscore)
    
    allauc +=auc_roc
    allpr += auc_pr    
    auclist.append(auc_roc)
    prlist.append(auc_pr) 
    f1score +=f1_score(y_test.ravel(), y_pred,average='weighted')
    f1score2 +=f1_score(y_test.ravel(), y_pred,average='binary')
    mrr,mr= evalu(tedge, y_pred_proba, y_test.ravel())
    mrr2,mr2= evalu(tedge2, y_pred_proba, y_test.ravel())
    print('局部排名结果',mrr,mr)       
    lmrr += mrr
    lmr += mr
    lmrr2 += mrr2
    lmr2 += mr2        
    ########计算 Hist@topK #############
    PAD = 0
    scores_len = 0
    y_prob = np.array(y_pred_proba)
    y_true = np.array(y_test)
    posnub = int(np.sum(y_true))
    k_list=[10, 50, 100]
    scores = {'hits@'+str(k):[] for k in k_list}
    for y_ in  range(y_true.shape[0]):
      y_prob_neg = y_prob[posnub:] 
      if y_true[y_] != PAD:
          scores_len += 1.0
          y_prob_neg = np.hstack((y_prob[y_],y_prob_neg))
          p_sort = y_prob_neg.argsort()
          for k in k_list:
            topk = p_sort[-k:][::-1]
            scores['hits@' + str(k)].extend([1. if 0 in topk else 0.])
    scores = {k: np.mean(v) for k, v in scores.items()} 
    print(scores)
    hist10 +=scores['hits@10']
    hist50 +=scores['hits@50']
    hist100 +=scores['hits@100']        

    ########计算 MR MRR #############        
    MR = mean_rank(y_test, y_pred_proba)
    MRR = mean_reciprocal_ranks(y_test, y_pred_proba)
    AP = average_precision(y_test, y_pred_proba)
    print(MR,MRR,AP)
    allmr +=MR
    allmrr +=MRR

aucstd = np.std(np.array(auclist))
prstd = np.std(np.array(prlist))
    
print('#'*10,'%d次随机8:2划分模型准确性结果'%(loop+1),'#'*10)
auc_mean = allauc/(loop+1)
pr_mean = allpr/(loop+1)
f1_mean = f1score/(loop+1)
f2_mean = f1score2/(loop+1)
p_mean = prec/(loop+1)
r_mean = reca/(loop+1)
print('%s_%s_%s%s模型AUC、PR、F1、precision和recall:'%(args.mt,args.gt,args.ds,args.dn),\
      "%.5f"%auc_mean,'和', "%.5f"%pr_mean, "%.5f"%f1_mean, "%.5f"%f2_mean, "%.5f"%p_mean, "%.5f"%r_mean)
    
print('#'*10,'%d次随机8:2划分排名结果'%(loop+1),'#'*10)
all_mr = allmr/(loop+1)
all_mrr = allmrr/(loop+1)
print('%s_%s_%s%s模型MR和MRR:'%(args.mt,args.gt,args.ds,args.dn), "%.5f"%all_mr,'和', "%.5f"%all_mrr)    
l_mr = lmr/(loop+1)
l_mrr = lmrr/(loop+1)
print('%s_%s_%s%s模型局部排名结果MR和MRR:'%(args.mt,args.gt,args.ds,args.dn), "%.5f"%l_mr,'和', "%.5f"%l_mrr)   
l_mr2 = lmr2/(loop+1)
l_mrr2 = lmrr2/(loop+1)
print('%s_%s_%s%s模型局部排名结果2MR和MRR:'%(args.mt,args.gt,args.ds,args.dn), "%.5f"%l_mr2,'和', "%.5f"%l_mrr2) 
hit10 =hist10/(loop+1)
hit50 =hist50/(loop+1)
hit100 =hist100/(loop+1)
print('%s_%s_%s%s模型hist@10、50、100:'%(args.mt,args.gt,args.ds,args.dn), "%.5f"%hit10,'和', "%.5f"%hit50, "%.5f"%hit100)

result = []
result.append(auc_mean)
result.append(pr_mean)
result.append(f2_mean)
result.append(hit10)
result.append(hit50)
result.append(hit100)
result.append(all_mr)
result.append(all_mrr)
result.append(l_mr)
result.append(l_mr2)
result.append(l_mrr)
result.append(l_mrr2)
result.append(f1_mean)
result.append(aucstd)
result.append(prstd)

col =['auc','pr','f1','h10','h50','h100','mr','mrr','lmr','lmr2','lmrr','lmrr2','f1_w','aucstd','prstd']
result =np.array(result)
result2 = result.reshape((1,15))

df=pd.DataFrame(result2,columns=col)
df.to_csv('D:/bioinformatics/2_model/GCN_model/SSLGRDA/result/result_%s_%s_%s%s.csv'%(args.mt,args.gt,args.ds,args.dn))
  
    
    
    
    
    
    
    
    
    
    
