# -*- coding: utf-8 -*-
# author: Jeffery_Xeom
# e-mail: 820367595@qq.com
# datetime: 2022/3/12 22:04
# software: PyCharm Pro

"""
This script is a test for histogram match
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
import math

import hist_equ_11911116

from numpy import polyfit

# global variable definition
x_lim_all = (0, 255)
y_lim_all = (0, 10)
x_lim = x_lim_all[1] - x_lim_all[0]
y_lim = y_lim_all[1] - y_lim_all[0]
# mapping = np.zeros(256)
func = lambda x, n=2: x ** n


def B_nx(n, i, x):
    if i > n:
        return 0
    elif i == 0:
        return (1 - x) ** n
    elif i == 1:
        return n * x * ((1 - x) ** (n - 1))
    return B_nx(n - 1, i, x) * (1 - x) + B_nx(n - 1, i - 1, x) * x


def test6():
    """已经实现了从鼠标点 然后正弦曲线下移的动态操作"""
    # 先生成一群点，把点画出来
    fig, ax = plt.subplots()
    x = np.linspace(1, 255, 10)  # 点的横坐标，放在一个数组里面
    y = np.ones(10)
    y = np.array(list(map(lambda n: n / 10, y)))
    plt.xlim(0, 255)  # 坐标系x轴范围
    plt.ylim(0, 10)  # 坐标系y轴范围
    ax.scatter(x, y)  # 画成散点图
    # 忽略除以0的报错
    np.seterr(divide='ignore', invalid='ignore')
    print("点击鼠标左键    当前的x_lim_all:\t", x_lim_all, "\t\t当前的y_lim_all:\t", y_lim_all)

    def calc_period_points(x1, x2, y1, y2):
        k = (y2 - y1) / (x2 - x1)
        return np.linspace()

    def calc_all_points(x, y):
        if x[0] < 0:
            x[0] = 0
        if x[25] > 255:
            x[25] = 255
        for i in range(len(x) - 1):
            calc_period_points(x[i], x[i + 1], y[i], y[i + 1])

    # # 二阶曲线方程
    # def func_2(x, a, b, c):
    #     return a * np.power(x, 2) + b * x + c

    # def polynomial(x_, args):
    #     res = 0
    #     n = len(args) - 1
    #     for i, item in enumerate(args):
    #         res += item * func(x_, n - i)
    #     return res

    def polynomial(x_, args):
        res = np.zeros(len(x_))
        n = len(args) - 1
        for i, item in enumerate(args):
            res += np.array(list(map(lambda x__: item * x__ ** (n - i), x_)))

        return res

    # 鼠标点击事件  函数里面又绑定了一个鼠标移动事件，所以生成的效果是鼠标按下并且移动的时候
    def on_button_press(event):
        if event.button == 1:
            # global x_lim_all
            # global y_lim_all
            print("点击鼠标左键    当前的x_lim_all:\t", x_lim_all, "\t\t当前的y_lim_all:\t", y_lim_all)
        fig.canvas.mpl_connect('motion_notify_event', on_button_move)

    # on_button_move 鼠标移动事件
    def on_button_move(event, y=y):
        current_ax = event.inaxes

        if event.button == 2:  # 1、2、3分别代表鼠标的左键、中键、右键，我这里用的是鼠标中键，根据自己的喜好选择吧
            x_mouse, y_mouse = event.xdata, event.ydata  # 拿到鼠标当前的横纵坐标
            ind = 512  # 这里生成一个列表存储一下要移动的那个点
            temp = 512
            # 计算一下鼠标的位置和图上点的位置距离，如果距离很近就移动图上那个点
            for i in range(len(x)):
                # 计算一下距离 图上每个点都和鼠标计算一下距离
                d = np.sqrt(((x_mouse - x[i]) / 256) ** 2 + (y_mouse - y[i]) ** 2)
                if d < temp:
                    ind = i
                    temp = d
                # if d < 5:  # 这里设置一个阈值，如果距离很近，就把它添加到那个列表中去
                #     if ind:
                #         ind[0]
                #         ind.append(i)
                #     else:
                #         ind.append(i)

            if ind < 256:  # 如果ind里面有元素，说明当前鼠标的位置距离图上的一个点很近
                # 通过索引ind[0]去改变当前这个点的坐标，新坐标是当前鼠标的横纵坐标（这样给人的感觉就是这个点跟着鼠标动了）
                y[ind] = y_mouse
                x[ind] = x_mouse

                # fit_coefficients = polyfit(x, y, 5)

                # # 然后根据所有点拟合出来一个二次方程曲线
                # popt2, pcov2 = curve_fit(func_2, x, y)
                # a2 = popt2[0]
                # b2 = popt2[1]
                # c2 = popt2[2]
                # yvals2 = func_2(x, a2, b2, c2)

                # 拟合好了以后把曲线画出来
                ax.cla()
                print("中键   当前的x_lim_all:\t", x_lim_all, "\t\t当前的y_lim_all:\t", y_lim_all)
                current_ax.set(xlim=x_lim_all,
                               ylim=y_lim_all)
                ax.scatter(x, y)
                # current_ax.set(xlim=x_lim_all,
                #                ylim=y_lim_all)
                ax.plot(x, y, 'r')

                # xvals2 = np.linspace(0, 255, 256)
                # yvals2 = polynomial(xvals2, fit_coefficients)
                # print('yvals2')
                # print(yvals2)

                # ax.plot(xvals2, yvals2, 'g')

                # xvals3 = np.linspace(0, 1, 100)
                # temp = np.zeros([np.size(x), 2])
                # for i in range(np.size(x)):
                #     temp[i] = [x[i], y[i]]
                # #     print(i, temp[i])
                # print('temp')
                # print(temp)
                # xvals3, yvals3 = get_newxy(temp, xvals3)
                # ax.plot(xvals3, yvals3, 'g')
                # # ax.scatter(xvals3, yvals3)

                xvals4, yvals4 = get_all_points(x, y)
                # print('xvals4')
                # print(xvals4)
                # print('yvals4')
                # print(yvals4)
                ax.scatter(xvals4, yvals4)

                fig.canvas.draw_idle()  # 重新绘制整个图表，所以看到的就是鼠标移动点然后曲线也跟着在变动
                # print('yvals3')
                # print(yvals3)

                # normalization_fac = sum(yvals2) /25.6

                normalization_fac = sum(yvals4) / 25.6

                output_Image, output_Hist, input_Hist, output_Hist_db, input_Hist_db = hist_match_11911116(
                    image_origin[:, :, 0],
                    np.array(list(map(lambda x_: H * W * x_ / normalization_fac, yvals4))))
                # output_Image, output_Hist, input_Hist, output_Hist_db, input_Hist_db = hist_match_11911116(
                #     image_origin[:, :, 0], get_match())

                cv2.imshow('output image', output_Image)

    def on_button_release(event):
        fig.canvas.mpl_disconnect(fig.canvas.mpl_connect('motion_notify_event', on_button_move))  # 鼠标释放事件
        print(x)
        print(y)

    def enlarge(event):
        global x_lim_all
        global y_lim_all
        x, y = event.xdata, event.ydata  # 这个暂时没有用上
        current_ax = event.inaxes
        xmin, xmax = current_ax.get_xlim()
        ymin, ymax = current_ax.get_ylim()
        x_step1, x_step2 = (x - xmin) / 10, (xmax - x) / 10
        y_step1, y_step2 = (y - ymin) / 10, (ymax - y) / 10

        if event.button == "up":  #
            # 鼠标向上滚动，缩小坐标轴刻度范围，使得图形变大
            x_lim_all = (xmin + x_step1, xmax - x_step2)
            y_lim_all = (ymin + y_step1, ymax - y_step2)
            current_ax.set(xlim=x_lim_all,
                           ylim=y_lim_all)
        if event.button == "down":  #
            # 鼠标向下滚动，增加坐标轴刻度范围，使得图形变小
            x_lim_all = (xmin - x_step1, xmax + x_step2)
            y_lim_all = (ymin - y_step1, ymax + y_step2)
            current_ax.set(xlim=x_lim_all,
                           ylim=y_lim_all)

        fig.canvas.draw_idle()

    print("最下面的x_lim_all:\t", x_lim_all, "\t\t当前的y_lim_all:\t", y_lim_all)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
    fig.canvas.mpl_connect('button_press_event', on_button_press)  # 鼠标点击事件
    fig.canvas.mpl_connect('button_release_event', on_button_release)  # 鼠标松开
    fig.canvas.mpl_connect('scroll_event', enlarge)  # 鼠标滚轮滚动事件
    plt.show()


def get_match():
    match = np.concatenate((np.linspace(0, 7, 6 - 0),
                            np.linspace(7, 0.75, 24 - 6),
                            np.linspace(0.75, 0, 184 - 24),
                            np.linspace(0, 0.5, 200 - 184),
                            np.linspace(0.5, 0, 256 - 200)), axis=0)
    # print(match)
    return match / sum(match)


def get_value(p, canshu):
    sumx = 0.
    sumy = 0.
    length = len(p) - 1
    for i in range(0, len(p)):
        sumx += (B_nx(length, i, canshu) * p[i][0])
        sumy += (B_nx(length, i, canshu) * p[i][1])
    return sumx, sumy


def get_newxy(p, x):
    xx = [0] * len(x)
    yy = [0] * len(x)
    for i in range(0, len(x)):
        print('x[i]=', x[i])
        a, b = get_value(p, x[i])
        xx[i] = a
        yy[i] = b
        print('xx[i]=', xx[i])
    return xx, yy


def get_all_points(x, y):
    xx = np.linspace(0, 255, 256)
    yy = np.linspace(0, 255, 256)
    x_size = len(x)
    for i in range(x_size - 1):
        x_min = int(np.ceil(max(x[i], 0)))
        x_max = int(np.floor(min(x[i + 1], 255)))
        k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        for j in range(x_min, x_max + 1):
            yy[j] = k * (xx[j] - x[i]) + y[i]
    return xx, yy


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


def mag2db(x):
    if x == 0:
        x = 1
    return 20 * math.log(x, 10)


def hist_match_11911116(input_image, spec_hist):
    input_hist = calc_hist(input_image)

    src_h, src_w = input_image.shape

    # 256 indicates how many levels are here
    # dtype uint8 indicates how many levels in the end
    s = np.zeros(256, dtype=np.uint8)
    z = np.zeros(256, dtype=np.uint8)
    mapping = np.zeros(256, dtype=np.uint8)

    temp_1 = 0
    temp_2 = 0
    total_pixel_input = src_h * src_w
    total_pixel_specifc = sum(spec_hist)
    for i in range(s.size):
        temp_1 = temp_1 + input_hist[i]
        s[i] = round(255 / total_pixel_input * temp_1)
        temp_2 = temp_2 + spec_hist[i]
        z[i] = round(255 / total_pixel_specifc * temp_2)

    index = 0
    for i in range(s.size):
        temp = abs(int(s[i]) - int(z[index]))
        while index < 255 and temp >= abs(int(s[i]) - int(z[index + 1])):
            temp = abs(int(s[i]) - int(z[index + 1]))
            index = index + 1
        mapping[i] = z[index]

    output_image = np.zeros((src_h, src_w), dtype=np.uint8)

    for i in range(src_h):
        for j in range(src_w):
            output_image[i][j] = mapping[input_image[i][j]]

    output_hist = calc_hist(output_image)

    input_hist_db = list(map(mag2db, input_hist))
    output_hist_db = list(map(mag2db, output_hist))

    return output_image, output_hist, input_hist, output_hist_db, input_hist_db


if __name__ == '__main__':
    # notice that the image should in the same path as this script
    image_path = 'Q3_2.tif'
    # show the original image
    image_origin = cv2.imread(image_path)
    cv2.imshow('original image 1', image_origin)

    H, W, _ = image_origin.shape

    # test6()

    t_start = time.time()
    output_Image, output_Hist, input_Hist, output_Hist_db, input_Hist_db = hist_match_11911116(image_origin[:, :, 0],
                                                                                               calc_hist(
                                                                                                   cv2.imread('1.tif')[
                                                                                                   :, :, 0]))
    t_end = time.time()
    print('local histogram equalization time elapse: ' + str(t_end - t_start) + ' s')

    cv2.imshow('output image 1', output_Image)

    print(sum(np.array(list(map(lambda x: int(x * H * W), get_match())))))

    output_Image2, output_Hist2, input_Hist2, output_Hist_db2, input_Hist_db2 = hist_match_11911116(
        image_origin[:, :, 0], np.array(list(map(lambda x: x * H * W, get_match()))))
    cv2.imshow('output image 2', output_Image2)

    test, _, _ = hist_equ_11911116.hist_equ_11911116(image_origin[:, :, 0])
    cv2.imshow('test', test)

    cv2.imwrite('hist_match_Q3_2_11911116.tif', output_Image)

    plt.subplot(221)
    plt.stem(range(256), input_Hist)
    plt.subplot(222)
    plt.stem(range(256), output_Hist)
    plt.subplot(223)
    plt.stem(range(256), input_Hist_db)
    plt.subplot(224)
    plt.stem(range(256), output_Hist_db)

    plt.figure(2)
    plt.subplot(221)
    plt.stem(output_Hist2)
    plt.subplot(222)
    plt.stem(get_match())
    plt.subplot(223)
    plt.stem(np.array(list(map(lambda x, y: x - y, output_Hist, output_Hist2))))
    plt.show()

    while True:
        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    plt.close()
    cv2.destroyAllWindows()
