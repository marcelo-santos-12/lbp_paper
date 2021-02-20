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

def to_bin(aux_absolute):
    n_bits=3
    binary = [0, 0, 0]
    for j in range(n_bits): # Decimal to Binary
        binary[j] = 1 if (aux_absolute & 1) == 1 else 0
        aux_absolute = aux_absolute >> 1
    return binary

def _extended_local_binary_pattern(double[:, ::1] image,
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

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute Extended LBP
    cdef Py_ssize_t n_bits = 3 # number of sublayers
    cdef int j, aux_absolute
    layers_shape = (n_bits + 1, P)
    cdef signed char [:, ::1] layers_bin = np.zeros(layers_shape, dtype=np.int8)
    cdef signed char[::1] binary = np.zeros(n_bits, dtype=np.int8)
    cdef double[:, ::1] output_layer1 = np.zeros(output_shape, dtype=np.double) # Original LBP
    cdef double[:, ::1] output_layer2 = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_layer3 = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_layer4 = np.zeros(output_shape, dtype=np.double)
    cdef double[::1] layers_lbp = np.zeros(layers_shape[0], dtype=np.double)   
    cdef double abs_difference

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P):
                    if texture[i] - image[r, c] >= 0: # Original LBP
                        layers_bin[0, i] = 1
                    else:
                        layers_bin[0, i] = 0

                    abs_difference = fabs(texture[i] - image[r, c])

                    aux_absolute = <int>abs_difference if abs_difference < 7 else 7 # According the original article
                    
                    for j in range(n_bits): # Decimal to Binary
                        binary[j] = 1 if aux_absolute & 1 == 1 else 0
                        aux_absolute >>= 1

                    layers_bin[1, i] = binary[0]
                    layers_bin[2, i] = binary[1]
                    layers_bin[3, i] = binary[2]
                    
                layers_lbp[0] = 0 # Original LBP
                layers_lbp[1] = 0
                layers_lbp[2] = 0
                layers_lbp[3] = 0

                # method == b'var' --> Don't permissed
                
                # if method == b'uniform':
                if method == b'U' or method == b'N':
                    for j in range(n_bits + 1): # for each layer
                        # determine number of 0 - 1 changes
                        changes = 0
                        for i in range(P - 1):
                            changes += (layers_bin[j, i] \
                                        - layers_bin[j, i + 1]) != 0
                        
                        if method == b'N':

                            if changes <= 2:
                                # We have a uniform pattern
                                n_ones = 0  # determines the number of ones
                                first_one = -1  # position was the first one
                                first_zero = -1  # position of the first zero
                                for i in range(P):
                                    if layers_bin[j, i]:
                                        n_ones += 1
                                        if first_one == -1:
                                            first_one = i
                                    else:
                                        if first_zero == -1:
                                            first_zero = i
                                if n_ones == 0:
                                    layers_lbp[j] = 0
                                elif n_ones == P:
                                    layers_lbp[j] = P * (P - 1) + 1
                                else:
                                    if first_one == 0:
                                        rot_index = n_ones - first_zero
                                    else:
                                        rot_index = P - first_one
                                    layers_lbp[j] = 1 + (n_ones - 1) * P + rot_index
                            else:  # changes > 2
                                layers_lbp[j] = P * (P - 1) + 2

                        else:  # method != 'N'
                            for j in range(n_bits + 1): # for each layer
                                if changes <= 2:
                                    for i in range(P):
                                        layers_lbp[j] += layers_bin[j, i]
                                else:
                                    layers_lbp[j] = P + 1
                else:
                    # method == b'default'
                    for j in range(n_bits + 1):
                        for i in range(P):
                            layers_lbp[j] += layers_bin[j, i] * weights[i]

                    # method == b'ror'
                    if method == b'R':
                        for j in range(n_bits + 1):
                            # shift LBP P times to the right and get minimum value
                            rotation_chain[0] = <int>layers_lbp[j]
                            for i in range(1, P):
                                rotation_chain[i] = \
                                    _bit_rotate_right(rotation_chain[i - 1], P)
                            layers_lbp[j] = rotation_chain[0]
                            for i in range(1, P):
                                layers_lbp[j] = min(layers_lbp[j], rotation_chain[i])

                output_layer1[r, c] = layers_lbp[0]
                output_layer2[r, c] = layers_lbp[1]
                output_layer3[r, c] = layers_lbp[2]
                output_layer4[r, c] = layers_lbp[3]

    return np.asarray([output_layer1, output_layer2 , output_layer3, output_layer4])
