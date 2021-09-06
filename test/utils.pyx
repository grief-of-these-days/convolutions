# -*- coding: utf-8 -*-
# cython: language_level=3
#
import numpy as np
cimport cython
cimport numpy as np


cdef inline int fmax(int one, int two) nogil:
    return one if one > two else two

cdef inline int fmin(int one, int two) nogil:
    return one if one < two else two


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef filter_w_bilinear_trick (
    np.ndarray[np.uint8_t, ndim=3] img,
    np.ndarray[np.float_t, ndim=2] coeffs0,
    np.ndarray[np.float_t, ndim=3] coords0):
    '''
    Simulation of the filtering procedure with
    bilinear trick to test the generated kernels.
    '''
    
    cdef int x, y, i, j
    cdef int xi0, xi1, yi0, yi1
    cdef float x0, y0, wx, wy
    cdef np.ndarray[np.float_t, ndim=1] px = np.zeros (3);

    cdef np.ndarray[np.float_t, ndim=2] coeffs = np.reshape(coeffs0, (-1, 1))
    cdef np.ndarray[np.float_t, ndim=2] coords = np.reshape(coords0, (-1, 2))
    cdef np.ndarray[np.uint8_t, ndim=3] result = np.zeros_like(img)

    cdef unsigned char [:, :, :] in_view = img
    cdef unsigned char [:, :, :] out_view = result

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            px[0] = 0
            px[1] = 0
            px[2] = 0
            for i in range(coeffs.shape[0]):
                x0 = x + coords[i, 0]
                y0 = y + coords[i, 1]
                xi0 = fmax(0, fmin(img.shape[1] - 1, int(x0)))
                yi0 = fmax(0, fmin(img.shape[0] - 1, int(y0)))
                xi1 = fmax(0, fmin(img.shape[1] - 1, int(x0) + 1))
                yi1 = fmax(0, fmin(img.shape[0] - 1, int(y0) + 1))
                wx = x0 - xi0
                wy = y0 - yi0
                for j in range(3):
                    px[j] += coeffs[i] * (
                        (float(in_view[yi0, xi0, j]) * (1 - wx) + float(in_view[yi0, xi1, j]) * wx) * (1 - wy) +
                        (float(in_view[yi1, xi0, j]) * (1 - wx) + float(in_view[yi1, xi1, j]) * wx) * wy)

            out_view[y, x, 0] = <unsigned char>px[0]
            out_view[y, x, 1] = <unsigned char>px[1]
            out_view[y, x, 2] = <unsigned char>px[2]

    return result
