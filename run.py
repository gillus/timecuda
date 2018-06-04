import numpy as np
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer 
import matplotlib.pyplot as plt
import time
# extract cuda_distance function pointer in the shared object cuda_distance.so
def get_cuda_distance():
    dll = ctypes.CDLL('./cuda_distance.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_distance
    func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t]
    return func

# create __cuda_distance function with get_cuda_distance()
__cuda_distance = get_cuda_distance()

# python wrapper for __cuda_distance
def cuda_dist(a, c2, size1, size2, KL1,stride):
    a_p = a.ctypes.data_as(POINTER(c_float))

    c2_p = c2.ctypes.data_as(POINTER(c_float))

    __cuda_distance(a_p, c2_p, size1, size2, KL1, stride)

if __name__ == '__main__':
	size1=int(2)   
	size2=int(48)
	KL1=int(5)
	KL2=int(1)
	stride=int(1)
	a = np.ones((size1,size2)).astype('float32')
	c1 = (np.zeros((size1,size2))+1000.).astype('float32')
	c2 = (np.zeros((size1,size2))+1000.).astype('float32')
	a[0,:]=1.1
	a[1,:]=2.10	
#	a[0,:KL1]=1.0+np.arange(0,KL1)
	a[0,-KL1*3:-KL1*2]=1.0+np.arange(0,KL1)
	a[0,-KL1:]=1.0+np.arange(0,KL1)
	a[1,-KL1:]=2.2
	t1=time.time()
	cuda_dist(a, c2, size2, size1, KL1,stride)
	t2=time.time()
	print(t2-t1,c1.min())
	print(a.shape,a[0,:])
	print(c1.shape,c1[0,:])
	print(c2.shape,c2[0,:])

