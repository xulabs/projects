import sys
from operator import add
from pytom.basic.resolution import bandToAngstrom, getResolutionBandFromFSC,  write_fsc2Ascii
from pytom.basic.correlation import FSC2, determineResolution2
from pytom_volume import read,vol,paste,subvolume, mean, variance, fsc2_v
from pytom_numpy import vol2npy,npy2vol  
from pyspark import SparkContext, SparkConf


from ctypes import cdll 
import numpy as np

def hanning_taper(volume1, avg):
    
    import math
    
    sizex = volume1.sizeX()
    sizey = volume1.sizeY()
    sizez = volume1.sizeZ()
    cx0 = math.pi * 2.0 /(sizex - 1)
    cx1 = math.pi * 2.0 /(sizey - 1)
    cx2 = math.pi * 2.0 /(sizez - 1)
    volume2 = vol(sizex,sizey,sizez)
    volume2.setAll(0)
 
    for zz in xrange(0, sizez):
        han2 = 0.5 * (1.0 - math.cos(cx2 * zz))
        for yy in xrange(0, sizey):
            han1 = 0.5 * (1.0 - math.cos(cx1 * yy))
            for xx in xrange(0, sizex):
               han0 = 0.5 * (1.0 - math.cos(cx0 * xx))
               vol_han = han0 * han1 * han2
               value2 = (volume1(xx,yy,zz) - avg)*vol_han + avg
               volume2.setV(round(value2,6),xx,yy,zz)

    del cx0,cx1,cx2,sizex,sizey,sizez,han0,han1,han2,value2,vol_han
    return volume2
    

def split_matrix(parId, iterator):
        
    x0 = -1
    y0 = -1
    z0 = -1
    t0 =  0 
    for x in iterator:
        x0 = x[1][0]
        y0 = x[1][1]
        z0 = x[1][2]
        t0 += 1
    
    
    vol10 = npy2vol(BC_vol1.value, 3)
    vol20 = npy2vol(BC_vol2.value, 3)
    boxhalf_x1 = BC_sizeX.value / 2 
    boxhalf_y1 = BC_sizeY.value / 2 
    boxhalf_z1 = BC_sizeZ.value / 2 
    vol1_sub = subvolume(vol10, x0 - boxhalf_x1, y0 - boxhalf_y1, z0 - boxhalf_z1, BC_sizeX.value, BC_sizeY.value, BC_sizeZ.value)
    vol2_sub = subvolume(vol20, x0 - boxhalf_x1, y0 - boxhalf_y1, z0 - boxhalf_z1, BC_sizeX.value, BC_sizeY.value, BC_sizeZ.value)
    
    
    avg1 = mean(vol1_sub)
    vol1_sub1 = hanning_taper(vol1_sub, avg1)
    avg2 = mean(vol2_sub)
    vol2_sub1 = hanning_taper(vol2_sub, avg2)


    res = fsc2_v(vol1_sub1, vol2_sub1, BC_fsc_criterion.value, BC_max_resolution.value, BC_pixelSize.value, x0, y0, z0)
    tup = (round(res,6), x0, y0, z0)
     
    del avg1, avg2, vol1_sub1, vol2_sub1
    del vol1_sub, vol2_sub, res, vol10, vol20, boxhalf_x1, boxhalf_y1, boxhalf_z1, x0, y0, z0, t0
    
    return tup

def volume_value_index(volume1,xx,yy,zz):
    
    nx = volume1.shape[0]
    ny = volume1.shape[1]
    nz = volume1.shape[2]

    if (xx >= nx or yy >= ny or zz >= nz):
        value = 0
    else:
        value = volume1[xx][yy][zz]

    del nx,ny,nz
    return value


def interpolate_gaps(volume1, step):
    """
    v1 is 3d array,numpy

    """
    
    v1_nx = volume1.shape[0]
    v1_ny = volume1.shape[1] 
    v1_nz = volume1.shape[2]
    
    
    x1 = x2 = y1 = y2 = z1 = z2 = 0
    fx = fx1 = fy = fy1 = fz = fz1 = 0
    v01 = v02 = v03 = v04 = v05 = v06 = v07 = v08 = 0
    vn = 0


    for zz in xrange(0,v1_nz,1):
        if (zz % step) == 0:
            z1 = zz
            z2 = z1 + step
        fz = (zz - z1)*1.0/step
        fz1 = 1 - fz
        for yy in xrange(0,v1_ny,1):
            if (yy % step) == 0:
                y1 = yy
                y2 = y1 + step
            fy = (yy- y1)*1.0/step
            fy1 = 1 - fy
            for xx in xrange(0,v1_nx,1):
                if (xx % step) == 0:
                    x1 = xx
                    x2 = x1 + step
                    
                    v01 = volume_value_index(volume1,x1,y1,z1)                 
                    v02 = volume_value_index(volume1,x2,y1,z1)
                    v03 = volume_value_index(volume1,x1,y2,z1)
                    v04 = volume_value_index(volume1,x2,y2,z1)
                    v05 = volume_value_index(volume1,x1,y1,z2)
                    v06 = volume_value_index(volume1,x2,y1,z2)
                    v07 = volume_value_index(volume1,x1,y2,z2)
                    v08 = volume_value_index(volume1,x2,y2,z2)
                fx = (xx - x1)*1.0/step
                fx1 = 1 - fx
                vn = fx1 * fy1 * fz1 * v01
                vn += fx * fy1 * fz1 * v02
                vn += fx1 * fy * fz1 * v03
                vn += fx * fy * fz1 * v04
                vn += fx1 * fy1 * fz * v05
                vn += fx * fy1 * fz * v06
                vn += fx1 * fy * fz * v07
                vn += fx * fy * fz * v08
                volume1[xx][yy][zz] = vn
    del  x1,x2,y1,y2,z1,z2,fx,fx1,fy,fy1,fz,fz1,v01,v02,v03,v04,v05,v06,v07,v08,vn
    return volume1

def rescale_to_avg_std(volume1, nuavg, nustd):
    avg = mean(volume1)
    std = np.sqrt(variance(volume1, False))

    if (std > 1e-30):
        scale = nustd/std
    else: 
        scale = 1
    shift = nuavg - avg*scale
    
    x = volume1.sizeX()
    y = volume1.sizeY()
    z = volume1.sizeZ()
    
    for zz in xrange(0,z):
        for yy in xrange(0,y):
            for xx in xrange(0,x):
                val = volume1(xx,yy,zz)*scale + shift
                volume1.setV(round(val,6),xx,yy,zz) 
    
    return volume1


if __name__ == "__main__":

    
    L = 64 
    pixelSize = 1.32
    max_resolution = 10
    fsc_criterion=0.143
    window_size = L
    increment= 4 
    step = increment 
    numberBands = int(window_size/8)  
    sizeX = sizeY = sizeZ = window_size


    conf = SparkConf().setAppName("parallel local FSC").setMaster("local[*]") 
    sc = SparkContext(conf=conf)
    vol1File = "./emd_3802_half_map_1.map"
    vol2File = "./emd_3802_half_map_2.map"
    hm1 = read(vol1File) 
    hm2 = read(vol2File) 

    avg1 = mean(hm1)
    std1 = np.sqrt(variance(hm1, False))
    hm20 = rescale_to_avg_std(hm2, avg1, std1)

    
    nz = hm1.sizeZ()
    ny = hm1.sizeY()
    nx = hm1.sizeX()

    mask_level = -1
    vz = vy = vx = window_size/2 
    pmask = vol(nx,ny,nz)
    pmask.setAll(1)
    
    # set up the mask for which voxels to calculate
    total_num = 0 
    for zz in xrange(0,nz):
        for yy in xrange(0,ny):
            for xx in xrange(0,nx):
                dovox = 1
                if (zz%step or yy%step or xx%step):
                    dovox = 0
                if (pmask(xx,yy,zz) < 0.5):
                    dovox = 0
                if (zz < vz or zz > (nz-vz)):
                    dovox = 0
                if (yy < vy or yy > (ny-vy)):
                    dovox = 0
                if (xx < vx or xx > (nx-vx)):
                    dovox = 0
                if ((mask_level >0) and (abs(pmask(xx,yy,zz)-mask_level) >0.1)):
                    dovox = 0
                pmask.setV(dovox,xx,yy,zz)

    boxhalf_x = boxhalf_y = boxhalf_z = window_size/2 
    nvox = pid = 0
    pid_matrix = []
    for zz in xrange(0,nz):
        for yy in xrange(0,ny):
            for xx in xrange(0,nx):
                if (pmask(xx,yy,zz)):
                    nvox+=1
                    pid += 1
                    key_value = (pid,(xx,yy,zz))
                    pid_matrix.append(key_value) 
    
    del nvox

     
    vol1_n = vol2npy(hm1) #must transform into 3D matrix
    vol2_n = vol2npy(hm20)
    vol1_np = np.zeros((nx,ny,nz),dtype=np.float32,order='F') 
    vol2_np = np.zeros((nx,ny,nz),dtype=np.float32,order='F') 
    vol1_np = vol1_n    
    vol2_np = vol2_n   
    

    BC_vol1 = sc.broadcast(vol1_np) 
    BC_vol2 = sc.broadcast(vol2_np) 

    BC_max_resolution = sc.broadcast(max_resolution)
    BC_pixelSize = sc.broadcast(pixelSize) 
    BC_numberBands = sc.broadcast(numberBands) 
    BC_fsc_criterion = sc.broadcast(fsc_criterion) 
    BC_sizeX = sc.broadcast(sizeX) 
    BC_sizeY = sc.broadcast(sizeY) 
    BC_sizeZ = sc.broadcast(sizeZ) 


    # distribute a local python collection to from an RDD
    rdd1 = sc.parallelize(pid_matrix).cache()
    
    rdd1_p = rdd1.partitionBy(pid)

    rddxy12 = rdd1_p.mapPartitionsWithIndex(split_matrix)
       
    volA_res_n = np.zeros((nx,ny,nz),dtype=np.float32,order='F') 

   
    matr_s = rddxy12.collect() 
    

    id0=0
    for i in xrange(0,len(matr_s), 4):
        i1 = int(matr_s[i+1])
        i2 = int(matr_s[i+2])
        i3 = int(matr_s[i+3])
        id0 +=1
        volA_res_n[i1][i2][i3] = matr_s[i]
    del matr_s
    

    if (step>1):
        volA_res_new = interpolate_gaps(volA_res_n,step)

    destination = './validation/'
    
    

    #write mrc file
    import mrcfile
    m_file = destination+'/'+'voltoA_3802_map.mrc'   
    
    with mrcfile.new(m_file, data=None, compression=None, overwrite=True) as mrc:
        mrc.set_data(volA_res_new)
        mrc.voxel_size = pixelSize 
        mrc.data
    del BC_vol1, BC_vol2, BC_pixelSize, BC_numberBands, BC_fsc_criterion, BC_sizeX, BC_sizeY, BC_sizeZ 
    del vol1_n, vol2_n, hm1, hm2, hm20, volA_res_new, volA_res_n 
