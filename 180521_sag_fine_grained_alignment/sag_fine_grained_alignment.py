'''
Created on May 21, 2018

@author: yongchunlu
'''

import numpy as np
import swig_frm 
from vol2sf import vol2sf


def transform_single_vol(vol1, mask):
    from pytom_volume import sum
    from pytom.basic.correlation import meanValueUnderMask, stdValueUnderMask, meanUnderMask, stdUnderMask
    
    p = sum(mask)
    meanV = meanUnderMask(vol1, mask, p)
    stdV = stdUnderMask(vol1, mask, p, meanV)
    result = (vol1 - meanV)/ stdV
    return result


def sag_fine_grained_alignment(vf, wf, vg, wg, max_freq, ang=[0,0,0], loc=[0,0,0], mask=None, B, alpha, maxIter, lambda1):
    """SAG-based fine-grained alignment between experimental data and reference data.
    Parameters
    vf: Experimental data
        pytom_volume.vol

    wf: Mask of vf in Fourier space.
        pytom.basic.structures.Wedge. If none, no missing wedge.

    vg: Reference data 
        pytom_volume.vol

    wg: Mask of vg in Fourier space.
        pytom.basic.structures.Wedge. If none, no missing wedge.

    max_freq: Maximal frequency involved in calculation.
              Integer.

    ang: Initial rotation angle

    loc: Initial translation value
    
    mask: Mask volume in real space.
          pytom_volume.vol

    B: Batch number 

    alpha: Step size

    maxIter: The max iteration number

    lambda1: Regularization parameter

    Returns
    -------
    (Optimal rotation angle and translation value.
    (best_translation, best_rotation, correlation_score)
    """
    from pytom_volume import vol, rotateSpline, peak, sum, power
    from pytom.basic.transformations import shift
    from pytom.basic.filter import lowpassFilter
    from pytom.basic.structures import Mask, SingleTiltWedge, Rotation
    from pytom_volume import initSphere
    from pytom_numpy import vol2npy

    import math
    import random
    

    if vf.sizeX()!=vg.sizeX() or vf.sizeY()!=vg.sizeY() or vf.sizeZ()!=vg.sizeZ():
        raise RuntimeError('Two volumes must have the same size!')

    if wf is None:
        wf = SingleTiltWedge(0)
    else: 
        vf = wf.apply(vf)
    if wg is None:
        wg = SingleTiltWedge(0)
    else: 
        vg = wg.apply(vg)

    if mask is None:
        m = vol(vf.sizeX(), vf.sizeY(), vf.sizeZ())
        initSphere(m, vf.sizeX()/2, 0,0, vf.sizeX()/2,vf.sizeY()/2,vf.sizeZ()/2)
        mask = m

    old_value = -1
    max_pos = [-1, -1, -1]
    max_ang = None
    max_value = -1.0

    ang_epsilon = np.ones(3) * (math.pi * (1.0/180))
    loc_epsilon = np.ones(3) * 1.0

    n = vf.sizeX()
    vf0_n = vol2npy(vf)

    if maxIter is None:
        maxIter = n/2
    iVals = np.int32(np.ceil((n-B) * np.random.random(maxIter))) 

    if lambda1 is None:
        lambda1 = 1/n
    eps = np.finfo(np.float32).eps 
    Lmax = 0.25*np.max(np.sum(vf0_n**2)) + lambda1 
    if alpha is None:
        alpha = 1/Lmax

    d = np.zeros(6) 
    g = np.zeros([n,6]) 
    covered = np.int32(np.zeros(n))
    nCovered = 0
    grad = np.zeros(6)
    deri = np.zeros(6)
    vg2 = vol(vf.sizeX(), vf.sizeY(), vf.sizeZ())
    mask2 = vol(mask.sizeX(), mask.sizeY(), mask.sizeZ())

    for i in range(n):
        if (covered[i]!=0 ):
            nCovered += 1

    for k in range(maxIter):
        i = iVals[k] -1 
        if k==0:
            rotateSpline(vg, vg2, ang[0], ang[1], ang[2])
            rotateSpline(mask, mask2, ang[0], ang[1], ang[2])
            
            vg2 = wf.apply(vg2) 
            vg2 = lowpassFilter(vg2, max_freq, max_freq/10.)[0]
            vg2_s = transform_single_vol(vg2, mask2) 

            vf2 = shift(vf, -loc[0]+vf.sizeX()/2, -loc[1]+vf.sizeY()/2, -loc[2]+vf.sizeZ()/2, imethod='spline')
            vf2 = lowpassFilter(vf2, max_freq, max_freq/10.)[0]
            vf2 = wg.apply(vf2, Rotation(ang))
            vf2_s = transform_single_vol(vf2, mask2)
            
            i = 0 
            ri = np.sum(vol2npy(vf2_s)[i:i+B,:,:] - vol2npy(vg2_s)[i:i+B,:,:]) 
            
        vg2_p  = vol(n,n,n)
        vg2_m  = vol(n,n,n)
        mask2_p = vol(n,n,n)
        mask2_m = vol(n,n,n)
        for dim_i in range(3):
            if abs(ang_epsilon[dim_i]) > eps:
                ang_epsilon_t = np.zeros(3)
                ang_epsilon_t[dim_i] = ang_epsilon[dim_i]

                angle = ang + ang_epsilon_t 
                rotateSpline(vg,vg2_p,angle[0],angle[1],angle[2]) 
                rotateSpline(mask,mask2_p,angle[0],angle[1],angle[2])
                vg2_p = wf.apply(vg2_p) 
                vg2_p = lowpassFilter(vg2_p, max_freq, max_freq/10.)[0]
                vg2_pf = transform_single_vol(vg2_p, mask2_p) 

                angle = ang - ang_epsilon_t
                rotateSpline(vg,vg2_m,angle[0],angle[1],angle[2]) 
                rotateSpline(mask,mask2_m,angle[0],angle[1],angle[2])
                vg2_m = wf.apply(vg2_m) 
                vg2_m = lowpassFilter(vg2_m, max_freq, max_freq/10.)[0]
                vg2_mf = transform_single_vol(vg2_m, mask2_m) 

                vg2_ang_deri = (vg2_pf - vg2_mf) / (2*ang_epsilon[dim_i])
                vg2_ang_deri_n = vol2npy(vg2_ang_deri)
                deri[dim_i] = np.sum(vg2_ang_deri_n[i:i+B,:,:])

                del vg2_pf, vg2_mf, vg2_ang_deri, vg2_ang_deri_n, angle
        del vg2_p, vg2_m, mask2_p, mask2_m
        
        vf1_ps = vol(n,n,n)
        vf1_ms = vol(n,n,n)
        ang_f =[ang[0], ang[1], ang[2]]
        for dim_i in range(3):
            if abs(loc_epsilon[dim_i]) > eps:
                loc_epsilon_t = np.zeros(3)
                loc_epsilon_t[dim_i] = ang_epsilon[dim_i]

                vf1_ps.copyVolume(vf)
                vf1_ms.copyVolume(vf)

                loc_r = loc + loc_epsilon_t 
                vf1_tp=shift(vf1_ps,-loc_r[0]+vf1_ps.sizeX()/2,-loc_r[1]+vf1_ps.sizeY()/2,-loc_r[2]+vf1_ps.sizeZ()/2, 'spline')
                vf1_tp = lowpassFilter(vf1_tp, max_freq, max_freq/10.)[0]
                vf1_tp = wg.apply(vf1_tp, Rotation(ang_f))

                loc_r = loc - loc_epsilon_t
                vf1_tm=shift(vf1_ms,-loc_r[0]+vf1_ms.sizeX()/2,-loc_r[1]+vf1_ms.sizeY()/2,-loc_r[2]+vf1_ms.sizeZ()/2, 'spline')
                vf1_tm = lowpassFilter(vf1_tm, max_freq, max_freq/10.)[0]
                vf1_tm = wg.apply(vf1_tm, Rotation(ang_f))

                vf1_tpf = transform_single_vol(vf1_tp,mask2)
                vf1_tmf = transform_single_vol(vf1_tm,mask2)
                vf1_loc_deri = (vf1_tpf - vf1_tmf) / (2*ang_epsilon[dim_i])
                vf1_loc_deri_n = vol2npy(vf1_loc_deri)
                deri[dim_i+3] = np.sum(vf1_loc_deri_n[i:i+B,:,:])

                del vf1_tp, vf1_tm, vf1_tpf, vf1_tmf, vf1_loc_deri, vf1_loc_deri_n
        del vf1_ps, vf1_ms, ang_f

        for dim_i in range(6):
            grad[dim_i] = ri*deri[dim_i]/B 

        for dim_i in range(6):
            d[dim_i] += grad[dim_i] - np.sum(g[i:i+B,dim_i])
            
        for dim_i in range(6):
            g[i:i+B,dim_i] = grad[dim_i] 

        for j0 in range(i,i+B+1): 
            if (covered[j0]==0):
                covered[j0] = 1
                nCovered +=1

        for dim_i in range(6):
            opt_beta[dim_i] -= alpha*d[dim_i]/nCovered
        ang = opt_beta[:3]
        loc = opt_beta[3:]
        
        rotateSpline(vg, vg2, ang[0], ang[1], ang[2])
        rotateSpline(mask, mask2, ang[0], ang[1], ang[2])
        
        vg2 = wf.apply(vg2) 
        vg2 = lowpassFilter(vg2, max_freq, max_freq/10.)[0]
        vg2_s = transform_single_vol(vg2, mask2) 

        vf2 = shift(vf, -loc[0]+vf.sizeX()/2, -loc[1]+vf.sizeY()/2, -loc[2]+vf.sizeZ()/2, imethod='spline')
        vf2 = lowpassFilter(vf2, max_freq, max_freq/10.)[0]
        vf2 = wg.apply(vf2, Rotation(ang))
        vf2_s = transform_single_vol(vf2, mask2)
        ri = np.sum(vol2npy(vf2_s)[i:i+B,:,:] - vol2npy(vg2_s)[i:i+B,:,:]) 
        from pytom.basic.correlation import nxcc
        val = nxcc(vf2_s, vg2_s, mask)
        
        if val > max_value:
            max_pos = loc
            max_ang = ang
            max_value = val
        if abs(max_value - old_value) <= eps:
            break
        else:
            old_value = max_value
        del vg2_s, vf2, vf2_s
        
    del d, g, grad,deri

    return max_pos, max_ang, max_value


