from itertools import product
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, fabs
from .interpolation cimport bilinear_interpolation, round
from cpython cimport array

ctypedef fused np_ints:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_uints:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

cdef inline int _bit_rotate_right(int value, int length) nogil:
    return (value >> 1) | ((value & 1) << (length - 1))

cdef signed char[:, ::1] uniforms(int P):
    def is_uniform(signed_texture, P):
            changes = 0
            for i in range(P - 1):
                changes += (signed_texture[i] - signed_texture[i + 1]) != 0
            return changes <= 2

    gen_comb = product([0, 1], repeat=P)
    cdef signed char[:, ::1] all_patterns_uniforms_char
    all_patterns_uniforms = [uniform for uniform in gen_comb if is_uniform(uniform, P)]
    all_patterns_uniforms_char = np.array(all_patterns_uniforms, dtype=np.int8)
    return all_patterns_uniforms_char

def _hamming_local_binary_pattern(double[:, ::1] image,
                          int P, float R, char method=b'U'):

    # texture weights
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)
    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cdef double[::1] rp = np.round(rr, 5)
    cdef double[::1] cp = np.round(cc, 5)

    # pre-allocate arrays for computation
    cdef double[::1] texture = np.zeros(P, dtype=np.double)
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    
    output_shape = (image.shape[0], image.shape[1])
    cdef double[:, ::1] output = np.zeros(output_shape, dtype=np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute Hamming LBP
    cdef Py_ssize_t n_uniforms = P * (P - 1) + 2
    cdef Py_ssize_t j, hamming_distance, hamming_distance_previous
    cdef Py_ssize_t dist_euclid, dist_euclid_previous
    cdef Py_ssize_t decimal_nonuniform, decimal_uniform
    cdef signed char[::1] results_signed_texture = np.zeros(P, dtype=np.int8)
    cdef signed char[:, ::1] all_uniforms = uniforms(P)
    
    with nogil:

        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P):
                    if texture[i] - image[r, c] >= 0:
                        signed_texture[i] = 1
                    else:
                        signed_texture[i] = 0

                lbp = 0

                # determine number of 0 - 1 changes
                changes = 0
                for i in range(P - 1):
                    changes += (signed_texture[i]
                                - signed_texture[i + 1]) != 0
                
                results_signed_texture = signed_texture
                
                # We dont have a uniform pattern
                if not changes <= 2:
                
                    hamming_distance_previous = P # max hamming distance
                    dist_euclid_previous = 2**P - 1 # max euclidian distance
                    decimal_nonuniform = 0
                    for i in range(P):
                        decimal_nonuniform += signed_texture[i] * weights[P - 1 - i]

                    for j in range(n_uniforms):
                        
                        hamming_distance = 0
                        for i in range(P): # hamming_distance
                            hamming_distance += (all_uniforms[j, i] - signed_texture[i]) != 0
                        
                        if hamming_distance <= hamming_distance_previous:
                            
                            decimal_uniform = 0
                            for i in range(P):
                                decimal_uniform += all_uniforms[j, i] * weights[P - 1 -i]
                            
                            dist_euclid = <Py_ssize_t>fabs(decimal_nonuniform - decimal_uniform)
                            
                            if dist_euclid < dist_euclid_previous:
                                
                                for i in range(P):
                                    results_signed_texture[i] = all_uniforms[j, i]
                                
                                # update minimum euclidian distance
                                dist_euclid_previous = dist_euclid
                            
                            # update minimum hamming distance
                            hamming_distance_previous = hamming_distance

                if method == b'N':    
                    n_ones = 0  # determines the number of ones
                    first_one = -1  # position was the first one
                    first_zero = -1  # position of the first zero
                    for i in range(P):
                        if results_signed_texture[i]:
                            n_ones += 1
                            if first_one == -1:
                                first_one = i
                        else:
                            if first_zero == -1:
                                first_zero = i
                    if n_ones == 0:
                        lbp = 0
                    elif n_ones == P:
                        lbp = P * (P - 1) + 1
                    else:
                        if first_one == 0:
                            rot_index = n_ones - first_zero
                        else:
                            rot_index = P - first_one
                        lbp = 1 + (n_ones - 1) * P + rot_index
                
                #method == b'U'
                else:

                    for i in range(P):            
                        lbp += results_signed_texture[i]
                
                output[r, c] = lbp

    return np.asarray(output)