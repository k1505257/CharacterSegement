import cv2
import math
import numpy as np
import copy
from sklearn.cluster import MeanShift


class Segment:


    def __init__(self, img_path):
        self.path = img_path
        self.img = cv2.imread(img_path)
        self.img = self.preprocessing(self.img)
        self.img_single = cv2.split(self.img)[0]
        (self.h, self.w) = self.img_single.shape
        #Y轴投影直方图
        self.distri_listY_raw = None
        # 原图与直方图的融合
        self.fusion = None
        # Y轴聚类群中心列表
        self.cluster_centers_axisY = None
        # Y轴聚类分割轴线图
        self.axisY_visual_mark = None
        # Y轴三合一:原图 直方图 中轴线
        self.img_line_axisY = None
        self.sections_axisY = []
        self.sections_axisX = []
        self.sections_axisX_temp = None

    #   二值化
    def threshold_demo(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # (b, g, r) =cv2.split(image)
        # gray = b
        ret, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        return binary

        #   泛洪填充

    def fill_color_demo(self, image):
        copy_image = image.copy()
        h, w = image.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        operater_down = (25, 25, 25)
        operater_up = (25, 25, 25)
        cv2.floodFill(copy_image, mask, (0, h - 1), (120, 160, 180),
                      operater_down, operater_up, cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(copy_image, mask, (w - 1, 0), (120, 160, 180),
                      operater_down, operater_up, cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(copy_image, mask, (0, 0), (120, 160, 180),
                      operater_down, operater_up, cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(copy_image, mask, (w - 1, h - 1), (120, 160, 180),
                      operater_down, operater_up, cv2.FLOODFILL_FIXED_RANGE)
        return copy_image

    # 腐蚀
    def dilate_demo(self, image, degree):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, degree)
        dst = cv2.erode(image, kernel)
        return dst

    # 膨胀
    def erode_demo(self, image, degree):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, degree)
        dst = cv2.dilate(image, kernel)
        return dst


    def projec2axisY(self):

        # 三通道转化为单通道
        img_simgle_copy_temp = copy.deepcopy(self.img_single)

        self.distri_listY_raw = [0 for temp_z in range(0, self.h)]
        for j in range(0, self.h):
            for i in range(0, self.w):
                if self.img_single[j, i] < 100:
                    self.distri_listY_raw[j] += 1
                    img_simgle_copy_temp[j, i] = 255

        for j in range(0, self.h):
            for i in range(0, self.distri_listY_raw[j]):
                img_simgle_copy_temp[j, i] = 0
        self.fusion = cv2.addWeighted(img_simgle_copy_temp, 0.3, cv2.split(self.img)[0], 0.7, 0)
        # cv2.imshow('fusion', self.fusion)
        # cv2.imshow('img', self.img_single)
        # cv2.imwrite('./visual/fusion_distriY.jpg', fusion)
        # cv2.waitKey()

    def cal_Gaussian(self, x, h=1):
        molecule = x * x
        denominator = 2 * h * h
        left = 1 / (math.sqrt(2 * math.pi) * h)
        return left * math.exp(-molecule / denominator)

    def meanshiftcluster_axisY(self, cicle, condense=1):
        # 半径cicle,参数压缩倍数condense
        self.distri_listY_use = copy.deepcopy(self.distri_listY_raw)

        # 高度转化为点的数目
        for i in range(len(self.distri_listY_raw)):
            self.distri_listY_use[i] = int(self.distri_listY_raw[i] / condense) * [i]

        # 二维展开成一维
        self.distri_listY_use = self.forfor(self.distri_listY_use)

        # 以为转化为二维，添加0
        for i in range(len(self.distri_listY_use)):
            self.distri_listY_use[i] = [self.distri_listY_use[i], 0]
        clustering = MeanShift(bandwidth=cicle).fit(self.distri_listY_use)
        self.cluster_centers_axisY = clustering.cluster_centers_

        # 可视化
        self.axisY_visual_mark = np.zeros((self.h, self.w), dtype="uint8")
        self.axisY_visual_mark[::] = 255
        for i in range(len(self.cluster_centers_axisY)):
            cv2.line(self.axisY_visual_mark, (0, int(self.cluster_centers_axisY[i][0])),
                     (self.w - 1, int(self.cluster_centers_axisY[i][0])), 0)

        self.img_line_axisY = cv2.bitwise_and(self.img_single, self.axisY_visual_mark)

        # 把乱序的self.cluster_centers_axisY从小到大排列,还是以(, )方式存在
        self.cluster_centers_axisY = np.sort(self.cluster_centers_axisY, axis=0)
        # print(self.cluster_centers_axisY)
        # cv2.imshow('axisY_visual_mark', self.img_line_axisY)
        # cv2.waitKey(0)

        # 二维列表展平

    def forfor(self, a):
        return [item for sublist in a for item in sublist]

    # 根据y轴聚类中心得到y区间
    def get_section_axisY(self, single_interval):
        self.sections_axisY = []
        for i in range(len(self.cluster_centers_axisY)):
            self.sections_axisY.append([self.cluster_centers_axisY[i][0] - single_interval,
                                        self.cluster_centers_axisY[i][0] + single_interval])

    def projec2axisX(self, section_start, section_end):
        section_end = int(section_end)
        section_start = int(section_start)
        w = self.img_single.shape[1]
        self.distri_listX = [0 for z in range(0, w)]

        img_simgle_copy_temp = copy.deepcopy(self.img_single)
        for j in range(0, w):  # 遍历一列
            for i in range(section_start, section_end):  # 遍历一行
                if img_simgle_copy_temp[i, j] == 0:  # 如果改点为黑点
                    self.distri_listX[j] += 1  # 该列的计数器加一计数
                    img_simgle_copy_temp[i][j] = 255
            # print (j)

        #
        for j in range(0, w):  # 遍历每一列
            for i in range((section_end - self.distri_listX[j]), section_end):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
                img_simgle_copy_temp[i, j] = 0  # 涂黑

        # 直方图可视化
        self.fusion_axisX = cv2.addWeighted(img_simgle_copy_temp, 0.3, cv2.split(self.img)[0], 0.7, 0)
        # cv2.imshow('fusion_axisX', self.fusion_axisX)
        # cv2.waitKey(0)

        # 区间划分
        self.sections_axisX_temp = []
        width_flag = 0  # 在突起上则为1
        for i in range(len(self.distri_listX)):
            if i == 0 or i == len(self.sections_axisX_temp) - 1:
                pass
            if self.distri_listX[i - 1] == 0 and self.distri_listX[i] > 0 and width_flag == 0:  # 凸起
                self.sections_axisX_temp.append(i)
                width_flag = 1
            if self.distri_listX[i - 1] > 0 and self.distri_listX[i] == 0 and width_flag == 1:  # 下降
                self.sections_axisX_temp.append(i)
                width_flag = 0

        # 保证为偶数
        if width_flag == 1:
            self.sections_axisX_temp.pop(len(self.sections_axisX_temp) - 1)

        # 一维转化为二维
        for i in range(0, len(self.sections_axisX_temp)):
            if i == len(self.sections_axisX_temp):
                break
            self.sections_axisX_temp[i] = [self.sections_axisX_temp[i], self.sections_axisX_temp[i + 1]]
            self.sections_axisX_temp.pop(i + 1)

        # print("sections_axisX_temp:{}".format(self.sections_axisX_temp))
        self.sections_axisX.append(self.sections_axisX_temp)

    def meanshiftcluster_axisX(self):
        # 每个区间运行一次
        for i in range(len(self.sections_axisY)):
            self.projec2axisX(self.sections_axisY[i][0], self.sections_axisY[i][1])


    def final_visualize(self, img):
        for i in range(len(self.sections_axisY)):
            for j in range(len(self.sections_axisX[i])):
                x1 = int(self.sections_axisX[i][j][0])
                y1 = int(self.sections_axisY[i][0])
                x2 = int(self.sections_axisX[i][j][1])
                y2 = int(self.sections_axisY[i][1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imwrite("./segment.jpg", img)
        cv2.imshow("3.segment", img)
        cv2.waitKey(0)


    # 图像预处理
    def preprocessing(self, img_pre):
        img_pre = self.fill_color_demo(img_pre)   # 角填充
        img_pre = self.threshold_demo(img_pre)    # 二值化
        cv2.imshow("1.binary", img_pre)

        return img_pre


if __name__ == '__main__':
    segment = Segment('./2.jpg')
    segment.projec2axisY()
    segment.meanshiftcluster_axisY(15, condense=100)
    segment.get_section_axisY(single_interval=8)
    segment.meanshiftcluster_axisX()

    img_save = cv2.imread(segment.path)
    segment.final_visualize(img_save)
