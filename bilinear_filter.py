#
# MIT License
#
# Copyright (c) 2021 https://github.com/grief-of-these-days
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import math
import numpy as np
import itertools
from types import SimpleNamespace


def bilinearKernel_row(k, c):
    """
    Helper function called by bilinearKernel()
    Performs a decomposition of a single kernel row.

    Args:
        k (np.array): Row of filter taps
        c (np.array): Rows of corresponding coords
    """
    assert len(k.shape) == 1
    assert (k.shape[0] % 2) != 0
    assert k.shape[0] == c.shape[0]

    result_filter = []
    result_coords = []
    r = k.shape[0]

    # Combine every two taps. Leave the center one to retain symmetry.
    range0 = [(x, x + 1) for x in range(0, r//2, 2)]
    range1 = [(x + 1, x) for x in range(r//2 + 1, r - 1, 2)]
    for col in itertools.chain (range0, range1):
        o1 = c[col[0]][0]
        o2 = c[col[1]][0]
        w1 = k[col[0]]
        w2 = k[col[1]]
        # Lerp between the coords.
        result_filter.append(w1 + w2)
        if (w1 + w2 > 0):
            result_coords.append([(o1 * w1 + o2 * w2) / (w1 + w2), (c[col[0]][1] + c[col[1]][1]) / 2])
        else:
            result_coords.append([0, 0])

    result_filter.insert (r//4, k[r//2])
    result_coords.insert (r//4, c[r//2].tolist())

    return SimpleNamespace(
        coeffs=np.array(result_filter),
        coords=np.array(result_coords))


def bilinearKernel(k):
    """
    This function converts an arbitrary (k,k) filter into a (k/2+1,k/2+1)
    kernel with bilinear lookup coords. This approach is useful to speed up
    the convolutions on gpu where bilinear lookups are virtually free by reducing
    the number of filter taps.
    See for example here: https://vec3.ca/bicubic-filtering-in-fewer-taps
                 or here: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/06/faster_filters.pdf

    Args:
        k (np.array): Array of filter taps of shape (k,k) where k is odd.

    Returns:
        Namespace of coeffs (reduced array of filter taps) and coords (array of their corresponding offsets).
    """
    assert len(k.shape) == 2
    assert k.shape[0] == k.shape[1]
    assert (k.shape[0] % 2) != 0

    r = k.shape[0]

    # Horizontal pass.
    result_filter = []
    result_coords = []

    for row in range(r):
        # Generate the colums coords for each row item and run the reduction pass.
        i = bilinearKernel_row(k[row], np.array(
            [(x, -(r//2) + row) for x in range(-(r//2), r//2 + 1)]).astype(float))
        result_filter.append(i.coeffs)
        result_coords.append(i.coords)

    # Transpose the reduced kernel.
    k1_t = np.transpose(result_filter)
    c1_t = np.transpose(np.flip(result_coords, axis=2), axes=[1, 0, 2])

    # Vertical pass.
    result_filter = []
    result_coords = []
    for row in range(r//2 + 1):
        i = bilinearKernel_row(k1_t[row], c1_t[row])
        result_filter.append(i.coeffs)
        result_coords.append(i.coords)

    # Transpose and return the final kernel.
    return SimpleNamespace(
        coeffs=np.transpose(result_filter),
        coords=np.transpose(np.flip(result_coords, axis=2), axes=[1, 0, 2]))


def test1():
    taps_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]).astype(float)
    taps_5x5 /= np.sum(taps_5x5)

    r = bilinearKernel(taps_5x5)
    print(r.coeffs)
    print(r.coords)
    

def test2():
    taps_9x9 = np.array([
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]).astype(float)
    taps_9x9 /= np.sum(taps_9x9)

    r = bilinearKernel(taps_9x9)
    print(r.coeffs)
    print(r.coords)


if __name__ == "__main__":
    ''' Sample usage '''

    test1()
    test2()
