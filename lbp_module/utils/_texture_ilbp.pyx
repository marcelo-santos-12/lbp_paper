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


def _improved_local_binary_pattern(double[:, ::1] image,
                          int P, float R, char method=b'D'):

    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cdef double[::1] rp = np.round(rr, 5)
    cdef double[::1] cp = np.round(cc, 5)
    cdef double[::1] texture = np.zeros(P, dtype=np.double)

    P += 1
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)

    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef double[:, ::1] output = np.zeros(output_shape, dtype=np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one
    
    # To compute the variance features
    cdef double sum_, var_, texture_i

    # To compute Improved LBP
    cdef double mean, texture_P

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P-1):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                
                mean = 0
                for i in range(P-1):
                    mean += texture[i]
                mean += image[r, c]
                mean /= P

                # verifing the neighboor pixels
                for i in range(P-1):                        
                    signed_texture[i] = 1 if texture[i] > mean else 0
                    
                # verifing the central pixel
                signed_texture[P-1] = 1 if image[r, c] > mean else 0
                
                lbp = 0
                        
                # if method == b'var':
                if method == b'V':
                    # Compute the variance without passing from numpy.
                    # Following the LBP paper, we're taking a biased estimate
                    # of the variance (ddof=0)
                    sum_ = 0.0
                    var_ = 0.0
                    for i in range(P-1):
                        texture_i = texture[i]
                        sum_ += texture_i
                        var_ += texture_i * texture_i
                    texture_P = image[r, c]
                    sum_ += texture_P
                    var_ += texture_P * texture_P
                    var_ = (var_ - (sum_ * sum_) / P) / P
                    if var_ != 0:
                        lbp = var_
                    else:
                        lbp = NAN
                # if method == b'uniform':
                elif method == b'U' or method == b'N':
                    # determine number of 0 - 1 changes
                    changes = 0
                    for i in range(P - 1):
                        changes += (signed_texture[i]
                                    - signed_texture[i + 1]) != 0
                    
                    if method == b'N':

                        if changes <= 2:
                            # We have a uniform pattern
                            n_ones = 0  # determines the number of ones
                            first_one = -1  # position was the first one
                            first_zero = -1  # position of the first zero
                            for i in range(P):
                                if signed_texture[i]:
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
                        else:  # changes > 2
                            lbp = P * (P - 1) + 2
                    else:  # method != 'N'
                        if changes <= 2:
                            for i in range(P):
                                lbp += signed_texture[i]
                        else:
                            lbp = P + 1
                else:
                    # method == b'default'
                    for i in range(P):
                        lbp += signed_texture[i] * weights[i]

                    # method == b'ror'
                    if method == b'R':
                        # shift LBP P times to the right and get minimum value
                        rotation_chain[0] = <int>lbp
                        for i in range(1, P):
                            rotation_chain[i] = \
                                _bit_rotate_right(rotation_chain[i - 1], P)
                        lbp = rotation_chain[0]
                        for i in range(1, P):
                            lbp = min(lbp, rotation_chain[i])

                output[r, c] = lbp

    return np.asarray(output)