#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:09:03 2019

@author: christoph
"""

import numpy as np

class Quaternion():
        
    def __init__(self,*args):
        if len(args)==2:
            self.inclination=args[0]
            self.azimuth=args[1]
            self.v=self.axisangle_to_q([np.cos(self.azimuth),np.sin(self.azimuth),0],self.inclination)
        if len(args)==4:
            self.v=np.array(args)
        
    def norm(self):
        return np.sqrt(np.sum(self.v**2))
        
    def normalize(self):
        self.v=self.v/self.norm()
           
    def q_mult(self, q2):
        (w1, x1, y1, z1) = self.v
        (w2, x2, y2, z2) = q2.v
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return Quaternion(w, x, y, z)
        
    def q_subtraction(self,q2):
        return Quaternion(self.v-q2.v)
        
    def q_addition(self,q2):
        return Quaternion(self.v+q2.v)

    def q_division(self,q2):
        pass
    
    def inverse(self):
        id_=1/np.sum(self.v**2)
        return Quaternion(id_*self.v[0],-id_*self.v[1],-id_*self.v[2],-id_*self.v[3])
        
    def q_conjugate(self):
        (w, x, y, z) = self.v
        return Quaternion(w, -x, -y, -z)
    
    def q_multConjugate(self,q2):
        return self.q_mult(q2.q_conjugate())
        
    
    ### this is the rotation of a list with 3D points 
    #       around this normalized Quaternion
    #   it converts every xyz to an Quaternion Q2 and
    #       new xyz= v1,v2,v3 of Q1*(Q2*conj(Q1))
    def qv_mult(self, v1):
        q2 = Quaternion(0,v1[0],v1[1],v1[2])
        newq=self.q_mult(q2.q_multConjugate(self))
        return np.array(newq.v[1:])
        
       
    def q_to_axisangle(self,q):
        (w, v) = (q[0], q[1:])
        theta = np.acos(w) * 2.0
        return ((v/np.sqrt(np.sum(v)**2)).tolist(), theta)


    def axisangle_to_q(self, v, theta):
        v = np.array(v)
        v=(v/np.sqrt(np.sum(v)**2)).tolist()
        (x, y, z) = v
        theta /= 2
        w = np.cos(theta)
        x = x * np.sin(theta)
        y = y * np.sin(theta)
        z = z * np.sin(theta)
        return np.array([w, x, y, z])

 

    