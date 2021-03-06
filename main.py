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
    keys=np.array(['blur','viewnumber','aberrations','contrast','intensities','positions','threefold',
                   'iterations','reconstruct','energy','potential','azimuth','stretch','smooth'])
    for key in keys:
        inputdic[key]=False
    inputdic['viewnumber']=-1
    inputdic['iterations']=1
    inputdic['energy']=0
    if not exists:
        print('writing input file')
        with open(path+'input.txt','w') as f:
            for key in keys:
                f.write('#'+key+'\tTrue\n')
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
    topfile='/home/christoph/gits/CH-reconstruction/tests3/noisy_projections.top' 
    #topfile='/home/christoph/gits/CH-reconstruction/tests2/noisy_projection.top'
    topfile='/home/christoph/gits/CH-reconstruction/tests_gb/gb18_41_prep.top'
    #topfile='/home/christoph/gits/CH-reconstruction/tests_small/noisy_projections.top' 
    path=os.path.dirname(topfile)+'/'
    inputdic={}
    read_inputFile(inputdic)
    image_stack=TiffFile(topfile[:-4]+'.tif').asarray()
    try:
        mask=TiffFile(path+'mask.tif').asarray()
    except FileNotFoundError:
        mask=[]
    print('initialize topology...')
    if inputdic.get('energy',0)>0:
        topfile2=path+str(inputdic['energy']).zfill(3)+'/'+os.path.basename(topfile)
        if not os.path.isfile(topfile2):
            os.makedirs(path+str(inputdic['energy']).zfill(3), exist_ok=True)
            from shutil import copy
            copy(topfile,path+str(inputdic['energy']).zfill(3)+'/'+os.path.basename(topfile))
        topfile=topfile2    
            
            
        
    print('using topfile '+topfile)
    master=Master.Master(topfile,path+'beamparameters.txt',image_stack,update_contrast=inputdic.get('contrast',False),reset_intensities=False,mask=mask)
    
    if inputdic.get('iterations',0)>0:
        while messagebox.askokcancel(title='Optimizer with energy '+str(inputdic.get('energy',0)), message='continue?'):
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
                if inputdic.get('energy',0)>0:
                    master.set_energy_contribution(inputdic.get('energy',0))
                    if master.lammps==None:
                        print('initializing lammps...')
                        master.init_lammps()
                if inputdic.get('azimuth',False):
                    print('matching azimuth angle...')
                    master.match_azimuth()  
                if inputdic.get('stretch',False):
                    print('matching stretch factor...')
                    master.update_stretch()  
                if inputdic.get('reconstruct',False):  
                    print('optimizing 3D structure...')
                    master.optimize_structure()                
                if inputdic.get('potential',False):  
                    print('calculating potential...')
                    master.calc_potentials()  
                if inputdic.get('smooth',False):  
                    print('smooth structure...')
                    master.smooth_structure()
                
            master.write_topfile()
            ar=master.save_errortrack()
            print('done!')
    #print('geometrical relaxation...')
    #master.equalize_bonds()
    if inputdic.get('energy',0)==0:
         master.init_lammps()
         master.update_structure()
         
    master.write_topfile()
    master.write_xyz()
    if (inputdic.get('positions',False) or inputdic.get('intensities',False) 
        or inputdic.get('reconstruct',False) or inputdic.get('contrast',False)):
        if messagebox.askokcancel(title='Optimizer', message='save new topology?'):
            master.save_topfile()
            
       
    
    print('Deviation per Atom: '+str(np.sqrt(master.update_disterror())))
    #master.write_topfile()
    
    #master.match_fov()
    #master.lammps.lmps.command('write_data new.data')
    
    #master.plot_intensities()
    #a.match_intensities()
    #master.update_plots()
    

    