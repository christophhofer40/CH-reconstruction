#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:04:06 2019

@author: christoph
"""
import numpy as np


def optimization(err_function, params,std=1,iterations=3,callback=None,mean=0):
    #print('old parameters:\t'+str(params))
    olderror=err_function(params)
    steparray=(np.zeros(len(params))+np.random.normal(loc=mean,scale=std))
    k=1
    for i in range(iterations):
        for i in range(len(params)):
            oldpar=params[i]
            params[i]+=steparray[i]
            newerror=err_function(params)
            if callback!=None:
                cbarray=[params[0],params[1],params[2],newerror,k]
                k+=1
                callback(cbarray)
            if newerror<olderror:
                olderror=newerror
                steparray[i]*=1.2
            else:
                params[i]=oldpar
                steparray*=-0.7
            
            
    #print('new parameters:\t'+str(params)+'\r\r')
    return params
        