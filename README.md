# timecuda
A Python library that can be used to calculate similarities in time series using GPU acceleration.

Given a multivariate time series, it calculates the Euclidean distance between the vector defined as the last N time steps and the rest of the series, back in time.

Original purpose was to use it to find the k-NNs of a time-series.

The GPU implentation is written in CUDA 9.1 and can be compiled with

nvcc -Xcompiler -fPIC -shared -o cuda_distance.so cuda_distance.cu

The file run.py contains the wrappers needed to use it as a Python function.
