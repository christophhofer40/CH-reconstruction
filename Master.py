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
from scipy.optimize import minimize
from shutil import move 
import random     
from Quaternion import Quaternion
from Run_lammps import Run_lammps
from Optimization import optimization as opt
lammps=None
energy_contribution=0

class Master:
    
    def __init__(self, topfile,beamfile,images,update_contrast=False,mask=[],reset_intensities=False):
        global globmaster
        self.topfile=topfile
        self.path=os.path.dirname(topfile)+'/'
        self.beamfile=beamfile
        self.image_stack=images
        self.atoms=[]
        self.views=[]
        self.rings=[]
        self.bonds=[]
        self.lammps=None
        self.mask=mask
        self.totalnumber=0
        self.viewnb=-1 #only optimize certain view
        self.init_from_topfile(reset_intensities)
        self.read_errortrack()
        self.init_beams()
        if update_contrast:
            self.update_contrast()
        
        self.disterror=-1
        self.update_structure()
        self.write_xyz()
        #self.energy=-1
        self.merit=-1
        self.errorlist=[]
        self.potentials=[]
        self.potentials1dx=[]
        self.potentials1dy=[]
        self.potentials1dz=[]
        
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
                        self.atoms.append(Atom(self,float(splitline[3]),float(splitline[4]),float(splitline[5]),int(splitline[1]),int(splitline[2])))
                        if len(splitline)>=7:
                            self.atoms[-1].element=int(splitline[6])
                        if len(self.image_stack.shape)>2:
                            if self.atoms[-1].viewcounts>=len(self.image_stack):
                                self.totalnumber+=1
                            else: #every atom gets accepted in one view
                                self.atoms[-1].member=False
                        else:
                            self.totalnumber+=1
                    else:
                        if int(splitline[1])>=0: 
                            if (math.isnan(self.image_stack[chapter][int(float(splitline[3])+0.5),int(float(splitline[2])+0.5)]) and 
                                self.atoms[int(splitline[1])].member):
                                #print('excluding atom id '+str(int(splitline[1])))
                                self.atoms[int(splitline[1])].member=False
                                self.totalnumber-=1

                        if math.isnan(float(splitline[2])) or math.isnan(float(splitline[3])):
                            if self.atoms[int(splitline[1])].member:                        
                                self.atoms[int(splitline[1])].member=False
                                self.totalnumber-=1
                            continue
                        twodatom=TwoDatom(float(splitline[2]),float(splitline[3]),int(splitline[1]),
                                  self.views[chapter],
                                  self.atoms[int(splitline[1])])
                        self.views[chapter].twoDatoms.append(twodatom)
                        self.atoms[int(splitline[1])].twodatoms.append(twodatom)
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
                    splitline=line.strip().split('\t')
                    a1=int(splitline[1])
                    a2=int(splitline[2])
                    if self.atoms[a1].member and self.atoms[a2].member:
                        self.bonds.append(Bond(self.atoms[a1],self.atoms[a2]))
                        self.atoms[a1].neighbors.append(self.atoms[a2])
                        self.atoms[a2].neighbors.append(self.atoms[a1])
                if line.startswith('RING'):
                    pass
                if line.startswith('QUATERNION'):
                    splitline=line.strip().split('\t')
                    self.views[chapter].quaternion=Quaternion(float(splitline[1]),float(splitline[2]))
        self.read_errortrack_3D()
        self.write_xyz()
            
    def init_beams(self):
        exists = os.path.isfile(self.beamfile)
        if exists:
            with open(self.beamfile,'r') as f:
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
            with open(self.beamfile,'w') as f:
                f.write('# reads out aberrations, field of view and sigma for gaussblur\n'+
                        '# Slicenumber	C10	C12a	C12b	C21a	C21b	C23a	C23b\n\n')
                for i in range(len(self.views)):
                    f.write('VIEW\t'+str(i)+'\n')
                    f.write('Aberrations\t0\t0\t0\t0\t0\t0\t0'+'\n') 
                    f.write('Sigma\t'+str(0)+'\n')
                    f.write('FieldOfView\t'+str(5)+'\n\n')       
 
    def init_lammps(self):
        global lammps
        lammps=Run_lammps(self.path,self)
        self.lammps=lammps
        
    def get_energy(self):
        pass
    
    def set_energy_contribution(self,energycontr):
        global energy_contribution
        energy_contribution=energycontr
        
    def write_beamparameters(self):
        with open(self.beamfile,'w') as f:
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
                
    def read_errortrack_3D(self):
        file=self.path+'error3D.txt'
        exists = os.path.isfile(file)
        if exists:
            self.errorlist=np.loadtxt(file).tolist()
            
    
    def write_topfile(self):
        with open(self.topfile[:-4]+'_new.top','w') as f:
            f.write('#ENERGY_CONTRIBUTION\t'+str(energy_contribution)+'\n')
            if lammps!=None:
                f.write('#ENERGY\t'+str(lammps.energy/self.totalnumber)+'\n')
            f.write('#DEVIATION\t'+str(self.disterror)+'\n\n')
            f.write('MASTER\t'+str(len(self.atoms))+'\n\n')
            for atom in self.atoms:
                f.write('ATOM\t'+str(atom.id)+'\t'+str(atom.viewcounts)+'\t'+str(atom.x)+'\t'+str(atom.y)+'\t'+str(atom.z)+'\t'+str(atom.element)+'\n')
            f.write('\n\n')
            for bond in self.bonds:
                f.write('BOND\t'+str(bond.a1.id)+'\t'+str(bond.a2.id)+'\n')
            for view in self.views:
                f.write('\n\nVIEW\t'+str(view.view)+'\n')
                f.write('QUATERNION\t'+str(view.quaternion.inclination)+'\t'+str(view.quaternion.azimuth)+'\n')
                f.write('TRANSLATION\t'+str(view.translation[0])+'\t'+str(view.translation[1])+'\n')
                f.write('BACKGROUND\t'+str(view.background)+'\n')
                f.write('GLOBAL_SCALE\t'+str(view.global_scale)+'\n')
                f.write('IMPURITIES\t'+str(view.impurities)+'\n\n')
                
                #f.write('VIEW\t'+str(view.view)+'\n')
                #f.write('VIEW\t'+str(view.view)+'\n')
                #f.write('VIEW\t'+str(view.view)+'\n')
                for atom2d in view.twoDatoms:
                    f.write('ATOM\t'+str(atom2d.id)+'\t'+str(atom2d.x)+'\t'+str(atom2d.y)+'\t'+str(atom2d.intensity)+'\n')
                    f.write('SEEN\t'+str(atom2d.id)+'\t'+str(atom2d.x_calc)+'\t'+str(atom2d.y_calc)+'\t'+str(atom2d.intensity)+'\n')
        np.savetxt(self.path+'error3D_new.txt',self.errorlist)        
                
    def save_topfile(self):
        self.write_topfile()
        move(self.topfile[:-4]+'_new.top',self.topfile)
        move(self.path+'error3D_new.txt',self.path+'error3D.txt')
       
        
    ## this method should be used to geomatrically relax the structure when there is only one view  
    def equalize_bonds(self):
        for it in range(20):
            for at in self.atoms:
                if at.member:
                    at.equalize_bonds()
            
                
                
    def match_threefold(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_threefold()
        self.write_beamparameters()
        #self.update_plots()
        
    def match_aberrations(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_aberrations()
        self.write_beamparameters()
        #self.update_plots()
        
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
        #self.update_plots()    
        
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
       # self.update_plots() 
        
    
    def optimize_2Dpositions(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.optimize_2Dpositions()
        self.write_topfile()
        #self.update_plots()    
        
    def update_contrast(self):
        for view in self.views:
            if view.view!=self.viewnb and self.viewnb!=-1:
                continue
            view.match_contrast()
        
        
    def update_disterror(self):
        self.disterror=0
        for view in self.views:
            self.disterror+=view.update_distance_error()
        self.disterror/=len(self.views)
        return self.disterror
    
    def update_disterror2(self): #same merit
        er=0
        for atom in self.atoms:
            if atom.member:
                er+=atom.get_error()
        er/=self.totalnumber
        return er
    
    def update_merit(self):
        if lammps!=None:
            energy=lammps.energy/self.totalnumber
        else:
            energy=0
        self.update_disterror()## average squared distance-deviation
        self.merit=self.disterror*(1-energy_contribution/100)+energy*energy_contribution/100 
        return self.merit
    
    def update_structure(self):
        for view in self.views:
            view.translation=[0,0]
        for atom in self.atoms:
            if not atom.member:
                continue
            atom.update_positions()
        for view in self.views:
            view.update_translation()
        self.update_merit() 
            
    def optimize_structure(self):
        for atom in self.atoms:
            if not atom.member:
                continue
            oldmerit=self.merit
            atom.optimize_position()
           # if lammps!=None:
            #    print('\nenergy:\t'+str(lammps.energy/self.totalnumber))
            self.update_merit()
            #print('disterror:\t'+str(self.disterror))
            if self.merit!=oldmerit:
                print('merit:\t'+str(self.merit))
        self.errorlist.append(self.merit)
        self.write_xyz()    
        self.write_topfile() 
                                  
    def write_xyz(self):
        scale=self.views[0].fov*10/self.views[0].imageWidth
        with open(self.path+'master.xyz','w') as f:
            f.write(str(self.totalnumber))
            f.write('\noptimized structure with energy contribution '+str(energy_contribution)+'\n')
            for atom in self.atoms:
                if not atom.member:
                    continue
                f.write(str(atom.element)+' '+str(atom.x*scale)+' '+str(atom.y*scale)+' '+str(atom.z*scale)+'\n')
                
     ### FOV of master is fov of first view
    def match_fov(self):
        def errfun(params):
            self.views[0].fov=params[0]
            lammps.update_positions()
            
            #print(str(lammps.energy))
            return lammps.energy
        #print('error before\t'+str(self.error))
        olderror=lammps.energy
        oldpar=self.views[0].fov
        teng,fopt=fmin(errfun,self.views[0].fov+random.normalvariate(0,0.5),maxiter=15,full_output=True,disp=True)[:2]
        if fopt>=olderror:
            #print('WARNING: optimization leads to larger error')
            self.views[0].fov=oldpar
            lammps.update_positions()
            print('fov not changed')
            return
        print('new fov: '+str(self.views[0].fov))
        self.views[0].fov=teng[0]
        lammps.update_positions()
        self.write_beamparameters()
        #print('error after\t'+str(self.error))
        
    def calc_potentials(self):
        if lammps==None:
            self.init_lammps()
        for atom in self.atoms:
            if not atom.member:
                continue
            atom.calc_potential()
        np.savetxt(self.path+'potential.txt',self.potentials,header='distance\tenergys')
        np.savetxt(self.path+'potential1dx.txt',self.potentials1dx,header='distance\tenergys')
        np.savetxt(self.path+'potential1dy.txt',self.potentials1dy,header='distance\tenergys')
        np.savetxt(self.path+'potential1dz.txt',self.potentials1dz,header='distance\tenergys')
    
            
                        
class Atom(Master):
    def __init__(self,master,x,y,z,id_,viewcounts,element=6):
        self.master=master
        self.x=x
        self.y=y
        self.z=z
        self.id=id_
        self.viewcounts=viewcounts
        self.element=element
        self.twodatoms=[]
        self.neighbors=[]
        self.rings=[]
        self.member=True #if it is included in 3D model
        
    def set_position(self,x,y,z,update_lammps=True):
        #global lammps
        self.x=x
        self.y=y
        self.z=z
        self.update_positions()
        if (lammps!=None and update_lammps):
            lammps.set_position(self.lammpsid,x,y,z)

        self.get_error()
        
    
    def update_positions(self):
        for atom in self.twodatoms:
            atom.update_position(updateError=True)
        
    def get_error(self):
        self.error=0
        k=0
        for atom in self.twodatoms:
            self.error+=atom.error
            k+=1
        if k!=0:
            self.error/=k
       
        return self.error
            
                
    def optimize_position(self):
         
         def errfun(param):
             self.set_position(param[0],param[1],param[2],update_lammps=(energy_contribution!=0))
#             if lammps!=None:
#                 energy=lammps.energy
#             else:
#                 energy=0
             
             return self.master.update_merit()
#         
#         if lammps!=None:
#             energy=lammps.energy
#         else:
#             energy=0
         def callbackF(Xi):
            #global Nfeval
            print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Xi[4], Xi[0], Xi[1], Xi[2], Xi[3]))
            #Nfeval += 1   
         
        
         olderror=self.master.update_merit()
         oldpars=[self.x,self.y,self.z]
#         (x,y,z),fopt=fmin(errfun,[self.x+random.normalvariate(0,2),
#                                            self.y+random.normalvariate(0,2),
#                                            self.z+random.normalvariate(0,2)],
#                                            maxiter=30,full_output=True,disp=False)[:2]
         #print(  '{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)')   )
         #print(  '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(0, self.x,self.y,self.z, self.master.merit)   )
#         res=minimize(errfun,[self.x+random.normalvariate(0,2),self.y+random.normalvariate(0,2),self.z+random.normalvariate(0,2)],
#                              options={'maxiter':1},callback=callbackF)
#         x=res.x[0]
#         y=res.x[1]
#         z=res.x[2]
         res=opt(errfun,oldpars,iterations=5,std=2)
         x=res[0]
         y=res[1]
         z=res[2]
         self.set_position(x,y,z,update_lammps=(energy_contribution!=0))
         fopt=self.master.update_merit()
         #print(  '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(1, self.x,self.y,self.z, self.master.merit)   )
         
         if fopt>=olderror:
             self.set_position(oldpars[0],oldpars[1],oldpars[2],update_lammps=(energy_contribution!=0))
             self.master.update_merit()
#         else:
#             self.set_position(x,y,z,update_lammps=(energy_contribution!=0))
    def calc_potential(self):
        def errfun(param):
             self.set_position(param[0],param[1],param[2],update_lammps=True)
        oldpars=[self.x,self.y,self.z] 
        olddist=self.master.disterror
        oldenergy=lammps.energy
        xs=random.normalvariate(0,2)
        ys=random.normalvariate(0,2)
        zs=random.normalvariate(0,2)
       
        ### "calibrate direction to energy     
        ## positive energy change corresponds to negative direction
        rets=[]#x,y,z
        vec=[xs,ys,zs]
        signs=[0,0,0]
        for i in range(3):
            ens=[]
            for sign in (1,-1):               
                newpos=oldpars.copy()
                newpos[i]+=(vec[i]*sign)
                self.set_position(newpos[0],newpos[1],newpos[2],update_lammps=True)
                self.master.update_disterror()
                ens.append((lammps.energy-oldenergy)/self.master.totalnumber)
            if ens[0]>ens[1]:
                signs[i]=-1
            else:
                signs[i]=1
            rets.append([abs(vec[i]/self.master.views[0].imageWidth*self.master.views[0].fov*10)*signs[i],self.master.disterror-olddist,ens[0]])
        self.set_position(oldpars[0]+xs,oldpars[1]+ys,oldpars[2]+zs,update_lammps=True)
        self.master.update_disterror()
        #radius=np.sqrt((oldpars[0]-xr)**2+(oldpars[1]-yr)**2+(oldpars[2]-zr)**2)/self.master.views[0].imageWidth*self.master.views[0].fov*1000
        ret=[abs(xs/self.master.views[0].imageWidth*self.master.views[0].fov*10)*signs[0],
             abs(ys/self.master.views[0].imageWidth*self.master.views[0].fov*10)*signs[1],
             abs(zs/self.master.views[0].imageWidth*self.master.views[0].fov*10)*signs[2],
             self.master.disterror-olddist,(lammps.energy-oldenergy)/self.master.totalnumber]
        self.master.potentials.append(ret)
        self.master.potentials1dx.append(rets[0])
        self.master.potentials1dy.append(rets[1])
        self.master.potentials1dz.append(rets[2])
        self.set_position(oldpars[0],oldpars[1],oldpars[2],update_lammps=True)
        self.master.update_disterror()
        return (ret,rets) #3D, array of each direction
    ## this method should be used to geomatrically relax the structure when there is only one view    
    def equalize_bonds(self):
        def errfun(param):
            self.z=param[0]
            return self.calc_bondstd()
        self.x=self.twodatoms[0].x
        self.y=self.twodatoms[0].y
        olderror=self.calc_bondstd()
        oldpar=self.z
        z,fopt=fmin(errfun,self.z+random.normalvariate(0,2),full_output=True,disp=False,maxiter=10)[:2]
        if fopt>=olderror:
            self.z=oldpar
        else:
            self.z=z[0]
            
    def calc_bondstd(self):
        atlist=[]
        scalefactor=np.sqrt(3) # next neigbor distance in a
        for at in self.neighbors:
            for at2 in at.neighbors:
                if at2!=self:
                    bondlength_scaled=np.sqrt((at2.x-self.x)**2+(at2.y-self.y)**2+(at2.z-self.z)**2)/scalefactor
                    atlist.append(bondlength_scaled)
            bondlength=np.sqrt((at.x-self.x)**2+(at.y-self.y)**2+(at.z-self.z)**2)
            atlist.append(bondlength)
        if len(atlist)==0:
            print('WARNING: No neighbors of atom number '+str(self.id))
            return 0
        return np.std(atlist)
   
class Bond:
    
    def __init__(self,a1,a2):
        self.a1=a1
        self.a2=a2
          

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
        self.quaternion=None
        self.translation=[10,0]
    
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
    
    def update_distance_error(self):
        self.disterror=0
        counts=0
        for atom in self.twoDatoms:
            if atom.atom.member:
                self.disterror+=atom.update_error()
                counts+=1
        if counts!=0:
            self.disterror/=counts
        return self.disterror
    
    def update_translation(self,apply=True):
        self.translation=[0,0]
        ct=0
        for atom in self.twoDatoms:
            if atom.atom.member:
                self.translation[0]+=atom.x-atom.x_calc
                self.translation[1]+=atom.y-atom.y_calc
                ct+=1
        if ct!=0:
            self.translation[0]/=ct
            self.translation[1]/=ct
        if apply:
            for atom in self.twoDatoms:
                if atom.atom.member:
                    atom.x_calc+=self.translation[0]
                    atom.y_calc+=self.translation[1]
                    

class TwoDatom(View):
    
    def __init__(self,x,y,id_,view,atom):
        self.x=x
        self.y=y
        self.id=id_
        self.view=view
        self.intensity=1 #scaling
        self.atom=atom
        self.x_calc=0
        self.y_calc=0
        self.error=-1
        
    def update_error(self):
        scale=1/self.view.imageWidth*self.view.fov*100#10 pm
        self.error=((self.x_calc-self.x)*scale)**2+((self.y_calc-self.y)*scale)**2
        return self.error
        
    def update_position(self,updateError=True):
        (self.x_calc,self.y_calc,self.z)=self.view.quaternion.qv_mult([self.atom.x,self.atom.y,self.atom.z])
        self.x_calc+=self.view.translation[0]
        self.y_calc+=self.view.translation[1]
        if updateError:
            self.update_error()