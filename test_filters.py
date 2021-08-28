import cv2 as cv
import numpy as np
from bilinear_filter import bilinearKernel


def filter_w_bilinear_trick(img, coeffs, coords):
    '''
    Simulation of the filtering procedure with
    bilinear trick to test the generated kernels.
    '''
    coeffs = np.reshape(coeffs, (-1, 1))
    coords = np.reshape(coords, (-1, 2))
    result = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            px = np.array([0.0, 0.0, 0.0])
            for i in range(coeffs.shape[0]):
                x0 = x + coords[i][0]
                y0 = y + coords[i][1]
                xi0 = max(0, min(img.shape[1] - 1, int(x0)))
                yi0 = max(0, min(img.shape[0] - 1, int(y0)))
                xi1 = max(0, min(img.shape[1] - 1, int(x0) + 1))
                yi1 = max(0, min(img.shape[0] - 1, int(y0) + 1))
                wx = x0 - xi0
                wy = y0 - yi0
                px += coeffs[i] * (
                    (img[yi0][xi0] * (1 - wx) + img[yi0][xi1] * wx) * (1 - wy) +
                    (img[yi1][xi0] * (1 - wx) + img[yi1][xi1] * wx) * wy)

            result[y][x] = px

    return result


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

    src = cv.imread('images/test1.jpg', cv.IMREAD_COLOR)
    dst_0 = cv.filter2D(src, -1, taps_5x5, borderType=cv.BORDER_REPLICATE)
    dst_1 = filter_w_bilinear_trick(src, r.coeffs, r.coords)

    d = np.mean(np.abs(dst_0 - dst_1))
    print("Avg L1 error is {}".format(d))
