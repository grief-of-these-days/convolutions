import sys
sys.path.append ('./test')

import cv2 as cv
import numpy as np
import scipy.signal as sig
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},
                                                reload_support=True)
from utils import filter_w_bilinear_trick
from bilinear_filter import bilinearKernel


def crop (img, sz):
    return img[sz:-sz, sz:-sz,:];


if __name__ == "__main__":
    taps= np.asarray([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]).astype(float)
    taps= (taps/ np.sum(taps))

    r = bilinearKernel(taps)

    src = cv.imread('./test/images/test1.jpg', cv.IMREAD_COLOR)
    dst_0 = np.stack ([
        sig.convolve2d (src[:,:,0], taps, mode="same", boundary="symm"),
        sig.convolve2d (src[:,:,1], taps, mode="same", boundary="symm"),
        sig.convolve2d (src[:,:,2], taps, mode="same", boundary="symm")
    ])
    dst_0 = np.transpose (dst_0, axes=[1, 2, 0])

    print("Testing generated filter...")
    dst_1 = filter_w_bilinear_trick(src, r.coeffs, r.coords)

    # Ignore boundaries.
    d = np.mean(np.abs(crop (dst_0, taps.shape[0]//2) - crop (dst_1, taps.shape[0]//2)))
    print("Avg L1 error is {}".format(d))
