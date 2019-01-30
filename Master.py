#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:28:35 2019

@author: christoph
"""
import numpy as np
from maptools import autotune
from scipy.ndimage.filters import gaussian_filter
import math
import os
from scipy.optimize import fmin
from shutil import move 
import random     



class Master:
    def __init__(self, topfile,beamfile,images,update_contrast=False,mask=[],reset_intensities=False):
        self.topfile=topfile
        self.path=os.path.dirname(topfile)+'/'
        self.beamfile=beamfile
        self.image_stack=images
        self.atoms=[]
        self.views=[]
        self.rings=[]
        self.bonds=[]
        self.mask=mask
        self.init_from_topfile(reset_intensities)
        self.read_errortrack()
        self.init_beams()
        if update_contrast:
            self.update_contrast()
        self.viewnb=-1 #only optimize certain view
        
        
    def init_from_topfile(self,reset_intensities):
        self.atoms=[]
        self.views=[]
        self.rings=[]
        self.bonds=[]
        with open(self.topfile,'r') as f:
            chapter=-1
            for line in f:
                if line.startswith('#') or line=='':
                    continue
                if line.startswith('ATOM'):
                    splitline=line.strip().split('\t')
                    if chapter==-1:
                        self.atoms.append(Atom(float(splitline[3]),float(splitline[4]),float(splitline[5]),int(splitline[1]),int(splitline[2])))
                        if len(splitline)>=7:
                            self.atoms[-1].element=int(splitline[6])
                    else:
                        if math.isnan(float(splitline[2])) or math.isnan(float(splitline[3])):
                            continue
                        self.views[chapter].twoDatoms.append(TwoDatom(float(splitline[2]),float(splitline[3]),int(splitline[1]),
                                  chapter,self.atoms[int(splitline[1])]))
                        if reset_intensities:
                            continue
                        if len(splitline)>=5:
                            self.views[chapter].twoDatoms[-1].intensity=float(splitline[4])
                            
                            
                if line.startswith('VIEW'):
                    chapter+=1
                    if (len(self.image_stack.shape)==2):
                        self.views.append(View(chapter,self.image_stack))
                    else:
                        self.views.append(View(chapter,self.image_stack[chapter]))
                    if self.mask!=[]:
                        self.views[chapter].mask=self.mask[chapter]
                
                if line.startswith('GLOBAL_SCALE'):
                    splitline=line.strip().split('\t')
                    self.views[chapter].global_scale=float(splitline[1])
                if line.startswith('BACKGROUND'):
                    splitline=line.strip().split('\t')
                    self.views[chapter].background=float(splitline[1])    
                if line.startswith('IMPURITIES'):
                    splitline=line.strip().split('\t')
                    self.views[chapter].impurities=float(splitline[1]) 
                    
                if line.startswith('BOND'):
                    pass
                if line.startswith('RING'):
                    pass
            
            
    def init_beams(self):
        exists = os.path.isfile(self.path+self.beamfile)
        if exists:
            with open(self.path+self.beamfile,'r') as f:
                for line in f:
                    if line.startswith('#') or line=='':
                        continue
                    splitline=line.strip().split('\t')
                    if splitline[0]=='VIEW':
                        chapter=int(splitline[1])
                    if splitline[0]=='Aberrations':
                        self.views[chapter].aberrations={
                                 'EHTFocus': float(splitline[1]), 
                                 'C12_a': float(splitline[2]), 
                                 'C12_b': float(splitline[3]), 
                                 'C21_a': float(splitline[4]),
                                 'C21_b': float(splitline[5]), 
                                 'C23_a': float(splitline[6]), 
                                 'C23_b': float(splitline[7])}
                    if splitline[0]=='Sigma':
                        self.views[chapter].sigma=float(splitline[1])
                    if splitline[0]=='FieldOfView':
                        self.views[chapter].fov=float(splitline[1])
        else:
            print('beamparameter file not found, initializing default values')
            with open(self.path+self.beamfile,'w') as f:
                f.write('# reads out aberrations, field of view and sigma for gaussblur\n'+
                        '# Slicenumber	C10	C12a	C12b	C21a	C21b	C23a	C23b\n\n')
                for i in range(len(self.views)):
                    f.write('VIEW\t'+str(i)+'\n')
                    f.write('Aberrations\t0\t0\t0\t0\t0\t0\t0'+'\n') 
                    f.write('Sigma\t'+str(0)+'\n')
                    f.write('FieldOfView\t'+str(5)+'\n\n')       
 
                    
                
        
    def write_beamparameters(self):
        with open(self.path+self.beamfile,'w') as f:
            f.write('# reads out aberrations, field of view and sigma for gaussblur\n'+
                        '# Slicenumber	C10	C12a	C12b	C21a	C21b	C23a	C23b\n\n') 
            for i in range(len(self.views)):                              
                f.write('VIEW\t'+str(i)+'\n')
                f.write('Aberrations\t'+str(self.views[i].aberrations.get('EHTFocus', 0))+
                        '\t'+str(self.views[i].aberrations.get('C12_a', 0))+
                        '\t'+str(self.views[i].aberrations.get('C12_b', 0))+'\t'+str(self.views[i].aberrations.get('C21_a', 0))+
                        '\t'+str(self.views[i].aberrations.get('C21_b', 0))+'\t'+str(self.views[i].aberrations.get('C23_a', 0))+
                        '\t'+str(self.views[i].aberrations.get('C23_b', 0))+'\n') 
                f.write('Sigma\t'+str(self.views[i].sigma)+'\n')
                f.write('FieldOfView\t'+str(self.views[i].fov)+'\n\n') 
    
    def write_topfile(self):
        with open(self.topfile[:-4]+'_new.top','w') as f:
            f.write('MASTER\t'+str(len(self.atoms))+'\n\n')
            for atom in self.atoms:
                f.write('ATOM\t'+str(atom.id)+'\t'+str(atom.viewcounts)+'\t'+str(atom.x)+'\t'+str(atom.y)+'\t'+str(atom.z)+'\t'+str(atom.element)+'\n')
            for view in self.views:
                f.write('\n\nVIEW\t'+str(view.view)+'\n')
                f.write('BACKGROUND\t'+str(view.background)+'\n')
                f.write('GLOBAL_SCALE\t'+str(view.global_scale)+'\n')
                f.write('IMPURITIES\t'+str(view.impurities)+'\n\n')
                #f.write('VIEW\t'+str(view.view)+'\n')
                #f.write('VIEW\t'+str(view.view)+'\n')
                #f.write('VIEW\t'+str(view.view)+'\n')
                for atom2d in view.twoDatoms:
                    f.write('ATOM\t'+str(atom2d.id)+'\t'+str(atom2d.x)+'\t'+str(atom2d.y)+'\t'+str(atom2d.intensity)+'\n')
                
    def save_topfile(self):
        self.write_topfile()
        move(self.topfile[:-4]+'_new.top',self.topfile)
                
                
    def match_threefold(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_threefold()
        self.write_beamparameters()
        self.update_plots()
        
    def match_aberrations(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_aberrations()
        self.write_beamparameters()
        self.update_plots()
        
    def reset_aberrations(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.aberrations={}

    def match_gaussblur(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_gaussblur()
        self.write_beamparameters()
        self.update_plots()    
        
    def save_errortrack(self):
        ar=[]
        for view in self.views:
            ar.append(view.track_error)
        np.save(self.path+'error',ar)
        return ar
    
    def read_errortrack(self):
        exists = os.path.isfile(self.path+'error.npy')
        if exists:
            ar=np.load(self.path+'error.npy').tolist()
            for view in self.views:
                view.track_error=ar[view.view]
    
    def match_intensities(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_intensities()
        self.write_topfile()
        self.update_plots() 
        
    
    def optimize_2Dpositions(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.optimize_2Dpositions()
        self.write_topfile()
        self.update_plots()    
        
    def update_contrast(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_contrast()
        
    
        
        
                            
                        
class Atom:
    def __init__(self,x,y,z,id_,viewcounts,element=6):
        self.x=x
        self.y=y
        self.z=z
        self.id=id_
        self.viewcounts=viewcounts
        self.element=element
    

        

class View:
    
    def __init__(self,view,image,**kwargs):
        self.twoDatoms=[]
        self.view=view
        self.image=np.array(image)
        self.error=-1
        self.aberrations = kwargs.get('aberrations', {})
        self.sigma=kwargs.get('sigma',2)
        self.fov=kwargs.get('fov',5)
        self.imageWidth=self.image.shape[1]
        self.imageHeight=self.image.shape[0]
        self.canvas_size=int(self.imageWidth/2) #enlargens deltalattice
        self.deltalattice=np.zeros((self.imageHeight+self.canvas_size-1,self.imageWidth+self.canvas_size-1))
        self.background=0  
        self.global_scale=1
        self.im=autotune.Imaging()
        self.track_error=[]
        self.diffimage=np.zeros((self.imageHeight,self.imageWidth))
        self.impurities=0
        self.mask=kwargs.get('mask',[])
        
    
    def simulate_image(self):
        self.get_deltalattice()
        cart_aberrations=self.polar2cartesian()
        self.simulation=self.im.image_grabber(delta_graphene=self.deltalattice,frame_parameters={'pixeltime':-1,'fov':self.fov,'size_pixels':tuple(self.image.shape)},
                    aberrations={'EHTFocus': self.aberrations.get('EHTFocus', 0), 
                                 'C12_a': cart_aberrations[0], 
                                 'C12_b': cart_aberrations[1], 
                                 'C21_a': cart_aberrations[2],
                                 'C21_b': cart_aberrations[3], 
                                 'C23_a': cart_aberrations[4], 
                                 'C23_b': cart_aberrations[5]},relative_aberrations=False)
        
        self.simulation=gaussian_filter(self.simulation,self.sigma)
        #self.match_mean_and_std()
        self.simulation[np.isnan(self.image)]=np.nan
        
    def polar2cartesian(self):
        #a ... amplitude, b... angle
        c12a=np.cos(self.aberrations.get('C12_b', 0))*self.aberrations.get('C12_a', 0)
        c12b=np.sin(self.aberrations.get('C12_b', 0))*self.aberrations.get('C12_a', 0)
        c21a=np.cos(self.aberrations.get('C21_b', 0))*self.aberrations.get('C21_a', 0)
        c21b=np.sin(self.aberrations.get('C21_b', 0))*self.aberrations.get('C21_a', 0)
        c23a=np.cos(self.aberrations.get('C23_b', 0))*self.aberrations.get('C23_a', 0)
        c23b=np.sin(self.aberrations.get('C23_b', 0))*self.aberrations.get('C23_a', 0)
        return [c12a,c12b,c21a,c21b,c23a,c23b]
    
    def get_deltalattice(self):
        self.deltalattice.fill(self.background)
        for atom2d in self.twoDatoms:
            px=atom2d.x
            py=atom2d.y
            if int(px)<0 or int(px)>self.imageWidth or int(py)<0 or int(py)>self.imageHeight:
                continue
            intensity=atom2d.atom.element**1.6*atom2d.intensity*self.global_scale
            self.deltalattice[int(self.canvas_size/2)+int(py),int(self.canvas_size/2)+int(px)]+=(1-(px-int(px)))*(1-(py-int(py)))*intensity
            self.deltalattice[int(self.canvas_size/2)+1+int(py),int(self.canvas_size/2)+int(px)]+=(px-int(px))*(1-(py-int(py)))*intensity
            self.deltalattice[int(self.canvas_size/2)+(int(py)+1),int(self.canvas_size/2)+int(px)+1]+=(px-int(px))*(py-int(py))*intensity
            self.deltalattice[int(self.canvas_size/2)+(int(py)+1),int(self.canvas_size/2)+int(px)]+=(1-(px-int(px)))*(py-int(py))*intensity
        
    def update_error(self):
        self.diffimage=self.simulation-self.image
        self.error=np.sqrt(np.nansum((self.diffimage)**2))
    
    def simulate_and_update_error(self):
        self.simulate_image()
        self.update_error()
        return self.error
        
    def match_mean_and_std(self):
        self.simulation-=np.nanmean(self.simulation)
        self.simulation=self.simulation/np.nanstd(self.simulation)*np.nanstd(self.image)
        self.simulation+=np.nanmean(self.image)
        
    def match_aberrations(self):
        def errfun(aberrations):            
            self.aberrations={'EHTFocus': aberrations[0], 'C12_a': aberrations[1],
                              'C12_b': aberrations[2], 'C21_a': aberrations[3],'C21_b': aberrations[4],
                              'C23_a':aberrations[5],'C23_b': aberrations[6]}
            self.simulate_and_update_error()
            return self.error
        
            #for i in range(rounds):
        #    c0,c12a,c12b,c21a,c21b,c23a,c23b=optimization(errfun,[self.aberrations['EHTFocus'],self.aberrations['C12_a'],self.aberrations['C12_b'],
        #                     self.aberrations['C21_a'],self.aberrations['C21_b'],self.aberrations['C23_a'],self.aberrations['C23_b']],[0.5,0.5,0.5,50,50,50,50])
        olderror=self.error
        oldpars=self.aberrations.copy()
        (c0,c12a,c12b,c21a,c21b,c23a,c23b),fopt=fmin(errfun,[self.aberrations.get('EHTFocus', 0)+random.normalvariate(0,3.5),
                                                      self.aberrations.get('C12_a', 0)+random.normalvariate(0,3),
                                                      self.aberrations.get('C12_b', 0)+random.normalvariate(0,10),
                                                      self.aberrations.get('C21_a', 0)+random.normalvariate(0,100),
                                                      self.aberrations.get('C21_b', 0)+random.normalvariate(0,10),
                                                      self.aberrations.get('C23_a', 0)+random.normalvariate(0,100),
                                                      self.aberrations.get('C23_b', 0)+random.normalvariate(0,10)],maxiter=10,full_output=True,disp=False)[:2]
        if fopt>=olderror and olderror!=-1:
            #print('WARNING: optimization leads to larger error')
            self.aberrations=oldpars
            self.simulate_and_update_error()
            return
        self.aberrations={'EHTFocus': c0, 'C12_a': c12a, 'C12_b': c12b, 'C21_a': c21a,'C21_b': c21b,  'C23_a': c23a,'C23_b': c23b}                    
        self.simulate_and_update_error()
        print('new error:\t'+str(self.error))     
        self.track_error.append(self.error)
        
    def match_threefold(self):
        def errfun(params):
            self.aberrations['C23_a']=params[0]
            self.aberrations['C23_b']=params[1]
            self.aberrations['EHTFocus']=params[2]
            self.sigma=params[3]
            self.simulate_and_update_error()
            return self.error
        olderror=self.error
        oldpars=[self.aberrations['C23_a'],self.aberrations['C23_b'],self.aberrations['EHTFocus'],self.sigma]
        (c23a,c23b,foc,sigma),fopt=fmin(errfun,[self.aberrations.get('C23_a', 0)+random.normalvariate(0,100),
                                            self.aberrations.get('C23_b', 0)+random.normalvariate(0,20),
                                            self.aberrations.get('EHTFocus', 0)+random.normalvariate(0,4),
                                            self.sigma+random.normalvariate(0,2)],
                                            maxiter=100,full_output=True,disp=False)[:2]
        if fopt>=olderror and olderror!=-1:
            #print('WARNING: optimization leads to larger error')
            self.aberrations['C23_a']=oldpars[0]
            self.aberrations['C23_b']=oldpars[1]
            self.aberrations['EHTFocus']=oldpars[2]
            self.sigma=oldpars[3]
            self.simulate_and_update_error()
            return
        self.aberrations={'C23_a': c23a,'C23_b': c23b,'EHTFocus':foc}  
        self.sigma=sigma                  
        self.simulate_and_update_error()
        self.track_error.append(self.error)
        print('new error:\t'+str(self.error))
        
    def match_contrast(self):
        print('optimizing contrast...')
        def errfun(params):
            self.background=params[0]
            self.global_scale=params[1]
            return self.simulate_and_update_error()
            
        
        #self.background,self.scale=optimization(errfun,[self.background,self.global_scale],[0.001,2],iterations=10)
        olderror=self.error
        oldpars=(self.background,self.global_scale)
        (bg,gs),fopt=fmin(errfun,[self.background,self.global_scale],full_output=True)[:2]
        if fopt>=olderror and olderror!=-1:
            print('WARNING: optimization leads to larger error')
            (self.background,self.global_scale)=oldpars
            self.simulate_and_update_error()
            return
        self.background=bg
        self.global_scale=gs
        self.simulate_and_update_error()
        print('new error:\t'+str(self.error))
        self.track_error.append(self.error)    
    
    def match_gaussblur(self):
        def errfun(params):
            self.sigma=params[0]
            return self.simulate_and_update_error()
        #print('error before\t'+str(self.error))
        olderror=self.error
        oldpar=self.sigma
        sigma,fopt=fmin(errfun,random.normalvariate(0,1.5),maxiter=10,full_output=True,disp=False)[:2]
        if fopt>=olderror:
            #print('WARNING: optimization leads to larger error')
            self.sigma=oldpar
            self.simulate_and_update_error()
            return
        
        self.sigma=sigma[0]
        self.simulate_and_update_error()
        print('new error:\t'+str(self.error))
        #print('error after\t'+str(self.error))
        self.track_error.append(self.error) 
        
    def match_intensities(self):
        def errfun(params,atom):
            atom.intensity=params[0]
            return self.simulate_and_update_error()
        err_before=self.error
        for at in self.twoDatoms:
            olderr=self.error
            oldpar=at.intensity
            if np.isnan(self.image[int(at.y+0.5),int(at.x+0.5)]):
                continue
            inten,fopt=fmin(errfun,random.normalvariate(0,at.intensity/4),  args=(at,),full_output=True,disp=True)[:2]
            #print('olderror:\t'+str(olderr))
            #print('newerror:\t'+str(fopt))
            
            if fopt>=olderr:
            #print('WARNING: optimization leads to larger error')a
                self.intensity=oldpar
                self.simulate_and_update_error()
                continue
            self.error=fopt
            at.intensity=inten[0]
        self.simulate_and_update_error()
        if self.error<=err_before:
            self.track_error.append(self.error) 
            print('new error:\t'+str(self.error))
        else:
            print('no improvement')
            
    def optimize_2Dpositions(self):
        def errfun(params,atom):
            atom.x=params[0]
            atom.y=params[1]
            return self.simulate_and_update_error()
        err_before=self.error
        for at in self.twoDatoms:
            if np.isnan(self.image[int(at.y+0.5),int(at.x+0.5)]):
                continue
            olderror=self.error
            #print('\n\nolderror:'+str(olderror))
            oldpars=(at.x,at.y)
            (x,y),fopt=fmin(errfun,[at.x+random.normalvariate(0,1.5),at.y+random.normalvariate(0,1.5)], args=(at,),full_output=True,maxiter=5,disp=False)[:2]
            if fopt>=olderror and olderror!=-1:
                #print('WARNING: optimization leads to larger error')
                #print('olderr:\t'+str(olderror))
               # print('newerr:\t'+str(fopt))
                (at.x,at.y)=oldpars
                self.simulate_and_update_error()
                continue
            self.error=fopt
            at.x=x
            at.y=y
            print('new error:\t'+str(self.error))
        self.simulate_and_update_error()
       
        if self.error<=err_before:
            self.track_error.append(self.error) 
        else:
            print('no improvement')    
            
    def get_intensities(self):
        ar=[]
        for at in self.twoDatoms:   
            if np.isnan(self.image[int(at.y+0.5),int(at.x+0.5)]):
                continue
            if self.mask!=[]:
                if self.mask[int(at.y+0.5),int(at.x+0.5)]==0:
                    continue
            ar.append(at.intensity)
        ar=np.array(ar)
        self.impurities=int(self.impurities)
        if self.impurities==0:
            return ar
        else:
            ar=ar[ar.argsort()]
            return [ar[:-self.impurities],ar[-self.impurities:]]
    
        
    
        
class TwoDatom(View):
    
    def __init__(self,x,y,id_,view,atom):
        self.x=x
        self.y=y
        self.id=id_
        self.view=view
        self.fov=-1
        self.intensity=1 #scaling
        self.atom=atom
        

        