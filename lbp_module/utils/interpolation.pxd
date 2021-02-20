import numpy as np
cimport numpy as np
from libc.math cimport ceil, floor

ctypedef fused np_ints:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

ctypedef fused np_uints:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    np.float32_t
    np.float64_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

cdef inline Py_ssize_t round(np_floats r) nogil:
    return <Py_ssize_t>((r + 0.5) if (r > 0.0) else (r - 0.5))

cdef inline Py_ssize_t fmax(Py_ssize_t one, Py_ssize_t two) nogil:
    return one if one > two else two

cdef inline Py_ssize_t fmin(Py_ssize_t one, Py_ssize_t two) nogil:
    return one if one < two else two

ctypedef fused np_real_numeric_out:
    np_real_numeric

cdef inline void bilinear_interpolation(
        np_real_numeric* image, Py_ssize_t rows, Py_ssize_t cols,
        np_floats r, np_floats c, char mode, np_real_numeric cval,
        np_real_numeric_out* out) nogil:

    cdef np_floats dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = <long>ceil(r)
    maxc = <long>ceil(c)
    dr = r - minr
    dc = c - minc

    cdef np.float64_t top
    cdef np.float64_t bottom

    cdef np_real_numeric top_left = get_pixel2d(image, rows, cols, minr, minc, mode, cval)
    cdef np_real_numeric top_right = get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    cdef np_real_numeric bottom_left = get_pixel2d(image, rows, cols, maxr, minc, mode, cval)
    cdef np_real_numeric bottom_right = get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    out[0] = <np_real_numeric_out> ((1 - dr) * top + dr * bottom)


cdef inline np_real_numeric get_pixel2d(np_real_numeric* image,
                                        Py_ssize_t rows, Py_ssize_t cols,
                                        long r, long c, char mode,
                                        np_real_numeric cval) nogil:
    
    if mode == b'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols + c]
    else:
        return <np_real_numeric>(image[coord_map(rows, r, mode) * cols +
                                       coord_map(cols, c, mode)])

cdef inline Py_ssize_t coord_map(Py_ssize_t dim, long coord, char mode) nogil:
    """Wrap a coordinate, according to a given mode.
    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'S', 'R', 'E'}
        Whether to wrap, symmetric reflect, reflect or use the nearest
        coordinate if `coord` falls outside [0, dim).
    """
    cdef Py_ssize_t cmax = dim - 1
    if mode == b'S': # symmetric
        if coord < 0:
            coord = -coord - 1
        if coord > cmax:
            if <Py_ssize_t>(coord / dim) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % dim))
            else:
                return <Py_ssize_t>(coord % dim)
    elif mode == b'W': # wrap
        if coord < 0:
            return <Py_ssize_t>(cmax - ((-coord - 1) % dim))
        elif coord > cmax:
            return <Py_ssize_t>(coord % dim)
    elif mode == b'E': # edge
        if coord < 0:
            return 0
        elif coord > cmax:
            return cmax
    elif mode == b'R': # reflect (mirror)
        if dim == 1:
            return 0
        elif coord < 0:
            # How many times times does the coordinate wrap?
            if <Py_ssize_t>(-coord / cmax) % 2 != 0:
                return cmax - <Py_ssize_t>(-coord % cmax)
            else:
                return <Py_ssize_t>(-coord % cmax)
        elif coord > cmax:
            if <Py_ssize_t>(coord / cmax) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % cmax))
            else:
                return <Py_ssize_t>(coord % cmax)
    return coord
