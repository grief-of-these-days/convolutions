import sys
sys.path.append ('./test')

import cv2 as cv
import numpy as np
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},
                                                reload_support=True)
from utils import filter_w_bilinear_trick
from bilinear_filter import bilinearKernel


if __name__ == "__main__":
    taps_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]).astype(float)
    taps_5x5 = (taps_5x5 / np.sum(taps_5x5)).reshape((5, 5))

    r = bilinearKernel(taps_5x5)

    src = cv.imread('./test/images/test1.jpg', cv.IMREAD_COLOR)
    dst_0 = cv.filter2D(src, -1, taps_5x5, borderType=cv.BORDER_REPLICATE)

    print("Testing generated filter...")
    dst_1 = filter_w_bilinear_trick(src, r.coeffs, r.coords)

    d = np.mean(np.abs(dst_0 - dst_1))
    print("Avg L1 error is {}".format(d))
