# -*- coding: utf-8 -*-
# author: Jeffery_Xeom
# e-mail: 820367595@qq.com
# datetime: 2022/3/13 01:45
# software: PyCharm Pro

"""
This script is a test for reduce salt and pepper noise
the default depth of the image is 256, with type uint8
If the input image is not as the description above, there may occur unexpected errors
"""

import time
# np is used for pixel process
import numpy as np
# cv2 is used for reading and showing images
import cv2.cv2 as cv2
import math


def reduce_SAP_11911116(input_image, n_size=3):
    """

    :param input_image:
    :param n_size:
    :return:
    """

    src_h, src_w = input_image.shape

    output_image1 = np.zeros((src_h, src_w), dtype=np.uint8)
    output_image2 = np.zeros((src_h, src_w), dtype=np.uint8)

    if n_size % 2 == 1:
        d = int((n_size - 1) / 2)
    else:
        print('error while process with m_size')
        print('use default m_size = 3')
        n_size = 3
        d = int((n_size - 1) / 2)
    length = n_size * n_size
    for i in range(d, src_h - d - 1):
        if i % 10 == 0:
            print(i)
        for j in range(d, src_w - d - 1):
            output_image1[i][j] = sorted((input_image[i - d:i + d + 1, j - d: j + d + 1].reshape(-1)))[
                math.floor(length / 2)]
            output_image2[i][j] = (sum(input_image[i - d:i + d + 1, j - d: j + d + 1].reshape(-1)) / length).astype(
                np.uint8)

    return output_image1, output_image2


if __name__ == '__main__':
    # notice that the image should in the same path as this script
    image_path = 'Q3_4.tif'
    # show the original image
    image_origin = cv2.imread(image_path)
    cv2.imshow('original image 1', image_origin)

    t_start = time.time()

    output_Image1, output_Image2 = reduce_SAP_11911116(
        image_origin[:, :, 0], 5)

    t_end = time.time()
    print('median and average filtering time elapse: ' + str(t_end - t_start) + ' s')

    cv2.imshow('output image 1', output_Image1)
    cv2.imshow('output image 2', output_Image2)

    cv2.imwrite('reduce_SAP_Q3_4_11911116.tif', output_Image1)

    while True:
        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
