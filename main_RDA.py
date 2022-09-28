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
