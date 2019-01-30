#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:22:19 2019

@author: christoph
"""
import Master
from tifffile import TiffFile
import os
#import tkinter
from tkinter import messagebox
import numpy as np

def read_inputFile(inputdic):
    exists = os.path.isfile(path+'input.txt')
    if not exists:
        return
    keys=np.array(['blur','viewnumber','aberrations','intensities','positions','threefold','iterations'])
    for key in keys:
        inputdic[key]=False
    inputdic['viewnumber']=-1
    inputdic['iterations']=1
    with open(path+'input.txt','r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            splitline=line.strip().split('\t')
            if splitline[0] in keys:
                try:
                    k=int(splitline[1])
                except ValueError:
                    k=splitline[1]=='True'
                inputdic[keys[keys==splitline[0]][0]]=k
    print(inputdic)
                
if __name__=='__main__':  
    #topfile='/home/christoph/hdd/DATA/samples/GO/configurations_statistics/configurations/pair/models/new/pair_stack-masked2.top'
    #topfile='/home/christoph/samples/GO/configurations_statistics/configurations/monovacancy/models/Stack-masked.top'   
    topfile='/home/christoph/samples/GO/configurations_statistics/configurations/vacancy-substitution/models/Stack-vacncies-masked2.top'
    path=os.path.dirname(topfile)+'/'
    image_stack=TiffFile(topfile[:-4]+'.tif').asarray()
    try:
        mask=TiffFile(path+'mask.tif').asarray()
    except FileNotFoundError:
        mask=[]
    print('initialize topology...')
    
    master=Master.Master(topfile,'beamparameters.txt',image_stack,update_contrast=False,reset_intensities=False,mask=mask)
    
    #master.reset_aberrations()
 
    
    #master.write_beamparameters()
    #master.update_plots()
    inputdic={}
    read_inputFile(inputdic)
    if inputdic['iterations']>0:
        while messagebox.askokcancel(title='Optimizer', message='continue?'):
            read_inputFile(inputdic)
            for i in range(inputdic.get('iterations',1)):
                master.viewnb=inputdic.get('viewnumber',-1)
                if inputdic['blur']:
                    print('optimize gaussblur...')
                    master.match_gaussblur()      
                if inputdic['aberrations']:
                    print('optimize aberrations...')
                    master.match_aberrations()
                if inputdic['threefold']:
                    print('matching three-fold-astigmatism')
                    master.match_threefold()
                if inputdic['positions']:
                    print('optimizing 2D positions')
                    master.optimize_2Dpositions()
                if inputdic['intensities']:
                    print('optimizing individual intensities...')  
                    master.match_intensities()
               
                
            master.write_topfile()
            ar=master.save_errortrack()
    if inputdic['positions'] or inputdic['intensities']:
        if messagebox.askokcancel(title='Optimizer', message='save new topology?'):
            master.save_topfile()

    #master.plot_intensities()
    #a.match_intensities()
    #master.update_plots()
    

    