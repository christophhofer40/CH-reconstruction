#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:18:46 2019

@author: christoph
"""

import matplotlib.pyplot as plt
import Master
import numpy as np
from tifffile import TiffFile
import os
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

if __name__=='__main__': 
    topfile='/home/christoph/samples/GO/configurations_statistics/configurations/vacancy-substitution/models/Stack-vacncies-masked2.top'
    #topfile='/home/christoph/hdd/DATA/samples/GO/configurations_statistics/configurations/pair/models/new/pair_stack-masked2.top'
    #topfile='/home/christoph/samples/GO/configurations_statistics/configurations/monovacancy/models/Stack-masked.top'
    name=os.path.basename(topfile)
    path=os.path.dirname(topfile)+'/'
    image_stack=TiffFile(topfile[:-4]+'.tif').asarray()
    
    try:
        mask=TiffFile(path+'mask.tif').asarray()
    except FileNotFoundError:
        mask=[]
        
    print('initialize topology...')
    master=Master.Master(topfile,'beamparameters.txt',image_stack,mask=mask)
    
    f,ax=plt.subplots(len(master.views),4,figsize=(15,20))
    f.suptitle('Plot of '+name,fontsize=28)
    for i in range(len(master.views)):
        for j in range(3):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_axis_off()
            
        master.views[i].simulate_and_update_error()
        ax[i,0].imshow(image_stack[i][~np.isnan(image_stack[i]).all(axis=0)])
        ax[i,1].imshow(master.views[i].simulation[~np.isnan(master.views[i].simulation).all(axis=0)])
        ax[i,2].imshow(master.views[i].diffimage[~np.isnan(master.views[i].diffimage).all(axis=0)])
        ax[i,3].plot(master.views[i].track_error)
    plt.tight_layout()
    plt.show()
    
    f2,ax2=plt.subplots(1,len(master.views),figsize=(8.3,5))
    for i in range(len(master.views)):   
        ints=master.views[i].get_intensities()
        std=np.std(ints[0])
        ax2[i].hist(master.views[i].get_intensities(),bins=30)
        ax2[i].text(ax2[i].get_xlim()[1]/2+ax2[i].get_xlim()[0]/2,ax2[i].get_ylim()[1],r'$\sigma=$'+str(round(std,3)),
           verticalalignment='bottom',horizontalalignment='center',fontsize=20)
    plt.show()
    