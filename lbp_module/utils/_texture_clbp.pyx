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

cdef inline double mean(double[::1] abs_diference) nogil:
    cdef Py_ssize_t i
    cdef double mean = 0
    for i in range(abs_diference.shape[0]):
        mean += abs_diference[i]
    return mean / abs_diference.shape[0]

def _completed_local_binary_pattern(double[:, ::1] image,
                          int P, float R, char method=b'D'):

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
    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef double[:, ::1] output = np.zeros(output_shape, dtype=np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double clbp_s
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute the variance features
    cdef double sum_, var_, texture_i

    # To compute Completed LBP
    cdef double[::1] abs_diference = np.zeros(P, dtype=np.double)
    cdef double abs_mean, clbp_m
    cdef Py_ssize_t j, c_changes
    cdef double[:, ::1] output_clbp = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_center = np.zeros(output_shape, dtype=np.double)
    cdef signed char[::1] abs_signed_texture = np.zeros(P, dtype=np.int8)
    
    cdef int[::1] c_rotation_chain = np.zeros(P, dtype=np.int32)
    
    cdef double c_sum_, c_var_, c_texture_i

    # Obtaining the information of Magnitude and for the central pixel
    cdef double sum_magnitude = 0, mean_magnitude

    cdef double mean_gray_scale = np.mean(image)
    cdef double central_pixel
    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # codign the central pixel
                if image[r, c] >= mean_gray_scale:
                    output_center[r, c] = 1
                
                for i in range(P):
                    
                    sum_magnitude += fabs(texture[i] - image[r, c])
    
    mean_magnitude = sum_magnitude / (P * image.shape[0] * image.shape[1])

    
    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P):
                    # coding the signal
                    if texture[i] - image[r, c] >= 0:
                        signed_texture[i] = 1
                    else:
                        signed_texture[i] = 0
                
                    abs_diference[i] = fabs(texture[i] - image[r, c])
                
                    # thresholding the absolute diference
                    # coding the magnitude
                    if abs_diference[i] >= mean_magnitude :
                        abs_signed_texture[i] = 1
                    else:
                        abs_signed_texture[i] = 0

                clbp_m = 0
                clbp_s = 0

                # if method == b'var':
                if method == b'V':
                    # Compute the variance without passing from numpy.
                    # Following the LBP paper, we're taking a biased estimate
                    # of the variance (ddof=0)
                    sum_, c_sum_ = 0.0, 0.0
                    var_, c_var_ = 0.0, 0.0
                    for i in range(P):
                        texture_i = texture[i]
                        sum_ += texture_i
                        var_ += texture_i * texture_i

                        c_texture_i = abs_diference[i]
                        c_sum_ += c_texture_i
                        c_var_ += c_texture_i * c_texture_i

                    var_ = (var_ - (sum_ * sum_) / P) / P
                    if var_ != 0:
                        clbp_s = var_
                    else:
                        clbp_s = NAN

                    c_var_ = (c_var_ - (c_sum_ * c_sum_) / P) / P
                    if c_var_ != 0:
                        clbp_m = c_var_
                    else:
                        clbp_m = NAN

                # if method == b'uniform':
                elif method == b'U' or method == b'N':
                    # determine number of 0 - 1 changes
                    changes = 0
                    c_changes = 0
                    for i in range(P - 1):
                        changes += (signed_texture[i]
                                    - signed_texture[i + 1]) != 0
                        c_changes += (abs_signed_texture[i]
                                    - abs_signed_texture[i + 1]) != 0
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
                                clbp_s = 0
                            elif n_ones == P:
                                clbp_s = P * (P - 1) + 1
                            else:
                                if first_one == 0:
                                    rot_index = n_ones - first_zero
                                else:
                                    rot_index = P - first_one
                                clbp_s = 1 + (n_ones - 1) * P + rot_index
                        else:  # changes > 2
                            clbp_s = P * (P - 1) + 2
                        
                        if c_changes <= 2:
                            # We have a uniform pattern
                            n_ones = 0  # determines the number of ones
                            first_one = -1  # position was the first one
                            first_zero = -1  # position of the first zero
                            for i in range(P):
                                if abs_signed_texture[i]:
                                    n_ones += 1
                                    if first_one == -1:
                                        first_one = i
                                else:
                                    if first_zero == -1:
                                        first_zero = i
                            if n_ones == 0:
                                clbp_m = 0
                            elif n_ones == P:
                                clbp_m = P * (P - 1) + 1
                            else:
                                if first_one == 0:
                                    rot_index = n_ones - first_zero
                                else:
                                    rot_index = P - first_one
                                clbp_m = 1 + (n_ones - 1) * P + rot_index

                        else:
                            clbp_m = P * (P - 1) + 2
                    else:  # method != 'N'
                        if changes <= 2:
                            for i in range(P):
                                clbp_s += signed_texture[i]
                        else:
                            clbp_s = P + 1

                        if c_changes <= 2:
                            for i in range(P):
                                clbp_m += abs_signed_texture[i]
                        else:
                            clbp_m = P + 1
                else:
                    # method == b'default'
                    for i in range(P):
                        clbp_s += signed_texture[i] * weights[i]
                        clbp_m += abs_signed_texture[i] * weights[i]

                    # method == b'ror'
                    if method == b'R':
                        # shift LBP P times to the right and get minimum value
                        rotation_chain[0] = <int>clbp_s
                        c_rotation_chain[0] = <int>clbp_m
                        for i in range(1, P):
                            rotation_chain[i] = \
                                _bit_rotate_right(rotation_chain[i - 1], P)
                            c_rotation_chain[i] = \
                                _bit_rotate_right(c_rotation_chain[i - 1], P)
                            
                        clbp_s = rotation_chain[0]
                        clbp_m = c_rotation_chain[0]
                        
                        for i in range(1, P):
                            clbp_s = min(clbp_s, rotation_chain[i])
                            clbp_m = min(clbp_m, c_rotation_chain[i])

                output[r, c] = clbp_s
                output_clbp[r, c] = clbp_m

    return np.asarray([output, output_clbp, output_center])