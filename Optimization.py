#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:04:06 2019

@author: christoph
"""
import random
import numpy as np


def optimization(err_function, params, steparray,std=0.2,iterations=3):
    #print('old parameters:\t'+str(params))
    olderror=err_function(params)
    sign=random.choice((-1,1))
    steparray=np.array(steparray)
    steparray=(steparray*np.random.normal(loc=1,scale=std))*sign
    for i in range(iterations):
        for i in range(len(params)):
            oldpar=params[i]
            params[i]+=steparray[i]
            newerror=err_function(params)
            if newerror<olderror:
                olderror=newerror
                steparray[i]*=1.2
            else:
                params[i]=oldpar
                steparray*=-0.7
    #print('new parameters:\t'+str(params)+'\r\r')
    return params
        