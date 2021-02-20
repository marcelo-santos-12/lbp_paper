import numpy as np
cimport numpy as cnp
from cpython cimport array

cdef inline double[::1] roi_hist(double[:, ::1] roi, bins):
    cdef int i = 0
    cdef double[::1] hist = np.zeros(bins, dtype=np.double)
    cdef int indice
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            indice = <int>roi[i, j]
            hist[indice] += 1
    return hist

cpdef double[::1] histogram(double[:, ::1] lbp_features, int P, int block, int bins):

    cdef int  r, i, h = <int>(lbp_features.shape[0] / block)
    cdef int c, j, v = <int>(lbp_features.shape[1] / block)

    cdef double[::1] hist = np.zeros(block * block * bins, dtype=np.double)
    
    cdef double[:, ::1] roi = np.zeros((h, v), dtype=np.double)
   
    cdef int cont = 0
    
    for r in range(0, lbp_features.shape[0], h):

        for c in range(0, lbp_features.shape[1], v):
            roi = lbp_features[r:r+h, c:c+v]

            hist[cont * bins:<int>((cont + 1) * bins)] = roi_hist(roi, bins)
            
            cont += 1

    return np.asarray(hist)
