# -*- coding: utf-8 -*-
# author: Jeffery_Xeom
# e-mail: 820367595@qq.com
# datetime: 2022/3/12 23:26
# software: PyCharm Pro

"""
This script is a test for local histogram equalization
the default depth of the image is 256, with type uint8
If the input image is not as the description above, there may occur unexpected errors
"""

import time

# np is used for pixel process
import numpy as np
# cv2 is used for reading and showing images
import cv2.cv2 as cv2
# plt is used for illustrating histograms
import matplotlib.pyplot as plt

import hist_equ_11911116
import reduce_SAP_11911116


def calc_hist(input_image):
    """
    This sub function is used for histogram calculation of an image
    --variables
    input_image{dtype = list, shape = ('src_h', 'src_w')}: The input image
    Note that the only acceptable shape of the input image is TWO dimensions
    --return value
    image_hist{dtype = list, shape = (dtype(input_image.element), )}:
    The list that represents the histogram of the input image.
    Note that this is a one dimension list, the size of the list
    is as large as the size of the data type of the element of the input image.
    The value of the element of image_hist indicates how many pixels at the value of
    the index of the image_hist are in the input_image
    """

    src_h, src_w = input_image.shape

    # initialize image_hist
    image_hist = np.zeros(256, dtype=np.int32)

    # traverse all the pixels in the input image
    # the value of the image pixel will be the index of the image_hist
    # the at that index the value of image_hist will plus one
    for i in range(src_h):
        for j in range(src_w):
            image_hist[input_image[i][j]] = image_hist[input_image[i][j]] + 1

    return image_hist


def calc_value_for_local_center(image):
    plane_image = image.flatten()
    center = int((plane_image.size - 1) / 2)
    num = sum(i <= plane_image[center] for i in plane_image)

    return round(255 / plane_image.size * num)


def local_hist_equ_11911116(input_image, m_size=3):
    """

    :param input_image: An two dimension array. The image that needed process.
    :param m_size: An odd integer. Indicate the length of side of the neighborhood
    :return:
    """
    src_h, src_w = input_image.shape

    input_hist = calc_hist(input_image)

    output_image = np.zeros((src_h, src_w), dtype=np.uint8)

    if m_size % 2 == 1:
        d = int((m_size - 1) / 2)
    else:
        print('error while process with m_size')
        print('use default m_size = 3')
        m_size = 3
        d = int((m_size - 1) / 2)

    for i in range(d, src_h - d - 1):
        if i % 10 == 0:
            print(i)
        for j in range(d, src_w - d - 1):
            output_image[i][j] = calc_value_for_local_center(
                input_image[i - d:i + d + 1, j - d: j + d + 1])
    output_hist = calc_hist(output_image)

    return output_image, output_hist, input_hist


if __name__ == '__main__':
    # notice that the image should in the same path as this script
    image_path = 'Q3_3.tif'
    # show the original image
    image_origin = cv2.imread(image_path)
    cv2.imshow('original image 1', image_origin)

    hist_equ_image, _, _ = hist_equ_11911116.hist_equ_11911116(image_origin[:, :, 0])

    cv2.imshow('global histogram equalization', hist_equ_image)

    t_start = time.time()
    output_Image, output_Hist, input_Hist = local_hist_equ_11911116(
        image_origin[:, :, 0], 5)
    t_end = time.time()

    print('local histogram equalization time elapse: ' + str(t_end - t_start) + ' s')

    cv2.imshow('output image 1', output_Image)

    # final_image = reduce_SAP_11911116.reduce_SAP_11911116(output_Image, 3)
    # cv2.imshow('final image', final_image)

    cv2.imwrite('local_hist_equ_Q3_3_11911116.tif', output_Image)

    plt.subplot(211)
    plt.stem(range(256), input_Hist)
    plt.subplot(212)
    plt.stem(range(256), output_Hist)
    plt.show()



    while True:
        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    plt.close()
    cv2.destroyAllWindows()
