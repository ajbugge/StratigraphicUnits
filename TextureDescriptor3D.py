import math
from scipy.interpolate import RegularGridInterpolator as RGI
import scipy.io
import numpy as np
import sys
import time
from IPython.display import clear_output

def LBP_3D_mixmax_corner(seismic_volume):
    """ Scikit's uniform local binary pattern function for 3D matrices."""
    start = time.time()
    points=[]
    for x in range (-1, 2, 2):
        for y in range (-1, 2, 2):
            for z in range (-1, 2, 2):
                if x==0 and y==0 and z==0:
                    continue
                else:
                    point=[x, y, z]
                    points.append(point)
    samples=len(points)
    size_x = seismic_volume.shape[0]
    size_y = seismic_volume.shape[1]
    size_z = seismic_volume.shape[2]
    
    grid_spacing = (
        np.arange(size_x),
        np.arange(size_y),
        np.arange(size_z)
    )
    interpolator = RGI(grid_spacing, seismic_volume, bounds_error=False, fill_value=0)
    output = np.zeros(seismic_volume.shape, dtype=np.uint64)
    
    weights =  2**np.arange(samples, dtype=np.uint64)
    signed_texture = np.zeros(samples, dtype=np.int8)
    total = size_x*size_y*size_z

    count=0
    for x in range (1, size_x):
        for y in range (1, size_y):
            for z in range (1, size_z):
                center_value = seismic_volume[x,y,z]
                local_points = np.add(points , [x,y,z])
                sphere_values = interpolator(local_points)
                largest=np.max(sphere_values)
                smallest=np.min(sphere_values)
                sml=True
                lg=True 
                for i in range(samples):
                    if sphere_values[i] == largest and sphere_values[i] > center_value and lg==True:
                        signed_texture[i] = 1
                        lg=False
                    elif sphere_values[i] == smallest and sphere_values[i] < center_value and sml==True:
                        signed_texture[i] = 1
                        sml=False
                    else:
                        signed_texture[i] = 0
                for i in range(samples):
                    if signed_texture[i]:
                        output[x,y,z] += weights[i]  # this is the LBP value
                        test=output[x,y,z] + weights[i]  
                count=count+1
                if count % 1000000 == 0:
                    clear_output()
                    stop = time.time()
                    t=(stop-start)/60
                    print(np.int((count/total)*100), '%. Time elapsed:', t)
         
    texture_descriptor=output.copy()
    unq=np.unique(output)
    num=0
    for i in unq:
        texture_descriptor[texture_descriptor==i]=num
        num=num+1
    return texture_descriptor