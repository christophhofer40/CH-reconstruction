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
    keys=np.array(['blur','viewnumber','aberrations','intensities','positions','threefold','iterations','reconstruct','energy'])
    for key in keys:
        inputdic[key]=False
    inputdic['viewnumber']=-1
    inputdic['iterations']=0
    inputdic['energy']=0
    if not exists:
        print('writing input file')
        with open(path+'input.txt','w') as f:
            for key in keys:
                f.write('#'+key+'\t'+str(inputdic[key])+'\n')
        return
    
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
    #topfile='/home/christoph/samples/GO/configurations_statistics/configurations/vacancy-substitution/models/Stack-vacncies-masked2.top'
    topfile='/home/christoph/gits/CH-reconstruction/tests/noisy_projections.top'
    path=os.path.dirname(topfile)+'/'
    image_stack=TiffFile(topfile[:-4]+'.tif').asarray()
    try:
        mask=TiffFile(path+'mask.tif').asarray()
    except FileNotFoundError:
        mask=[]
    print('initialize topology...')
    
    master=Master.Master(topfile,'beamparameters.txt',image_stack,update_contrast=False,reset_intensities=False,mask=mask)
    
    inputdic={}
    read_inputFile(inputdic)
        
    if inputdic.get('iterations',0)>0:
        while messagebox.askokcancel(title='Optimizer', message='continue?'):
            read_inputFile(inputdic)
            for e in range(10):
                inputdic['energy']=e+1
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
                    if inputdic.get('energy',0)>0:
                        master.set_energy_contribution(inputdic.get('energy',0))
                        if master.lammps==None:
                            master.init_lammps()
                    if inputdic.get('reconstruct',False):  
                        print('optimizing 3D structure')
                        master.optimize_structure()
               
                
            master.write_topfile()
            ar=master.save_errortrack()
    if inputdic.get('positions',False) or inputdic.get('intensities',False) or inputdic.get('reconstruct',False):
        if messagebox.askokcancel(title='Optimizer', message='save new topology?'):
            master.save_topfile()
    
    print('Deviation per Atom: '+str(master.update_disterror()))
    #master.write_topfile()
    
    #master.init_lammps()
    #master.match_fov()
    #master.lammps.lmps.command('write_data new.data')
    master.write_xyz()
    #master.plot_intensities()
    #a.match_intensities()
    #master.update_plots()
    

    