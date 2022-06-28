# -*- coding: utf-8 -*-
# author: Jeffery_Xeom
# e-mail: 820367595@qq.com
# datetime: 2022/3/7 16:31
# software: PyCharm Pro

"""
This script is a test for histogram equalization
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


def calc_hist(input_image):
    """
    This sub function is used for histogram calculation of an image
    --variables
    input_image{dtype = np.array, shape = ('src_h', 'src_w')}: The input image
    Note that the only acceptable shape of the input image is TWO dimensions
    --return value
    image_hist{dtype = list, shape = (sizeOf(dtype(input_image.element))'the date type depth of the input element', )}:
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
    # at that index the value of image_hist will plus one
    for i in range(src_h):
        for j in range(src_w):
            image_hist[input_image[i][j]] += 1
    return image_hist


def hist_equ_11911116(input_image):
    """
    This function is used for general histogram equalization
    --variables
    input_image{dtype = np.array, shape = ('src_h', 'src_w')}: The input image
    Note that the only acceptable shape of the input image is TWO dimensions
    --return value
    output_image{dtype = np.array, shape = ('src_h', 'src_w')}: The equalized image
    output_hist and input_hist{dtype = np.array,
        shape = (sizeOf(dtype(input_image.element))'the date type depth of the input element', )}:
    The list that represents the histogram of the output and input image.
    Note that this is a one dimension list, the size of the list
    is as large as the size of the data type of the element of the input image.
    The value of the element of image_hist indicates how many pixels at the value of
    the index of the image_hist are in the input_image
    """
    # 256 indicates how many levels are here
    # dtype uint8 indicates how many levels in the end
    s = np.zeros(256, dtype=np.uint8)
    src_h, src_w = input_image.shape
    output_image = np.zeros((src_h, src_w), dtype=np.uint8)
    input_hist = calc_hist(input_image)

    temp = 0
    total_pixel = src_h * src_w
    for i in range(s.size):
        temp += input_hist[i]
        s[i] = round(255 / total_pixel * temp)

    for i in range(src_h):
        for j in range(src_w):
            output_image[i][j] = s[input_image[i][j]]
    output_hist = calc_hist(output_image)

    return output_image, output_hist, input_hist


if __name__ == '__main__':
    # notice that the image should in the same path as this script
    image_path_1 = 'Q3_1_1.tif'
    # show the original image
    image_origin_1 = cv2.imread(image_path_1)
    cv2.imshow('original image 1', image_origin_1)

    t_start = time.time()

    output_Image_1, output_Hist_1, input_Hist_1 = hist_equ_11911116(image_origin_1[:, :, 0])

    t_end = time.time()
    print('local histogram equalization time elapse: ' + str(t_end - t_start) + ' s')

    cv2.imshow('output image 1', output_Image_1)

    plt.subplot(221)
    plt.stem(range(256), input_Hist_1)
    plt.subplot(222)
    plt.stem(range(256), output_Hist_1)
    # this if for the second image
    image_path_2 = 'Q3_1_2.tif'
    # show the original image
    image_origin_2 = cv2.imread(image_path_2)
    cv2.imshow('original image 2', image_origin_2)

    output_Image_2, output_Hist_2, input_Hist_2 = hist_equ_11911116(image_origin_2[:, :, 0])

    cv2.imshow('output image 2', output_Image_2)

    cv2.imwrite('hist_equ_Q3_1_1_11911116.tif', output_Image_1)
    cv2.imwrite('hist_equ_Q3_1_2_11911116.tif', output_Image_2)

    plt.subplot(223)
    plt.stem(range(256), input_Hist_2)
    plt.subplot(224)
    plt.stem(range(256), output_Hist_2)
    plt.show()



    while True:
        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    plt.close()
    cv2.destroyAllWindows()
