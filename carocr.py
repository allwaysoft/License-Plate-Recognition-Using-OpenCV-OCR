import cv2
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
src = cv2.imread("lLD9016.jpg")  # 打开要识别的照片，不能有中文路径
print(src.shape)  # 输出粗一下原图的大小
license = cv2.resize(src, (800, int(800 * src.shape[0] / src.shape[1])))  # 压缩一下图片，保持了原图的宽高的比例
print(license.shape)  # 输出一下压缩过后的大小
cv2.namedWindow('inputImage', 0)  # 第二个参数为0，可以改变窗口的大小
# cv2.imshow('inputImage', src)
cv2.imshow('inputImage', license)
cv2.waitKey(0)


# cv2.destroyAllWindows()


def license_prepation(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 从RGB图像转为hsv色彩空间
    low_hsv = np.array([108, 43, 46])  # 设置颜色
    high_hsv = np.array([124, 255, 255])
    mask = cv2.inRange(image_hsv, lowerb=low_hsv, upperb=high_hsv)  # 选出蓝色的区域
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    image_dst = cv2.bitwise_and(image, image, mask=mask)  # 取frame与mask中不为0的相与，在原图中扣出蓝色的区域，mask=mask必须有
    cv2.imshow('license_dst', image_dst)
    cv2.waitKey(0)
    image_blur = cv2.GaussianBlur(image_dst, (7, 7), 0)  # 高斯模糊，消除噪声。第二个参数为卷积核大小，越大模糊的越厉害
    cv2.imshow('license_blur', image_blur)
    cv2.waitKey(0)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    ret, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
    cv2.imshow('binary', binary)
    cv2.waitKey(0)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 6))  # 得到一个4*6的卷积核
    image_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)  # 开操作，去一些干扰
    cv2.imshow('license_opened', image_opened)
    cv2.waitKey(0)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 得到一个7*7的卷积核
    image_closed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, kernel2)  # 闭操作，填充一些区域
    cv2.imshow('license_closed', image_closed)
    cv2.waitKey(0)
    return image_closed


license_prepared = license_prepation(license)
contours, hierarchy = cv2.findContours(license_prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def choose_license_area(contours, Min_Area):
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:  # 面积大于MIN_AREA的区域保留
            temp_contours.append(contour)
    license_area = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        # print(rect_tupple)
        rect_width, rect_height = rect_tupple[1]  # 0为中心点，1为长和宽，2为角度
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            license_area.append(temp_contour)
    return license_area


license_area = choose_license_area(contours, 2000)


def license_segment(license_area):
    if (len(license_area)) == 1:
        for car_plate in license_area:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)  # 行是row 列是col
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)  # 这两行代码为了找出车牌位置的坐标
            card_img = license[col_min:col_max, row_min:row_max, :]
            cv2.imshow("card_img", card_img)
            cv2.waitKey(0)
            cv2.imwrite("card_img.jpg", card_img)

    return card_img


result = license_segment(license_area)
cv2.imshow('result', result)  # 将检测到的车牌显示出来
cv2.waitKey(0)


def recognize_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)  # 二值化
    cv2.imshow('bin', binary)  # 显示二值过后的结果， 白底黑字
    cv2.waitKey(0)
    bin1 = cv2.resize(binary, (370, 82))  # 改变一下大小，有助于识别
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # 获取一个卷积核，参数都是自己调的
    dilated = cv2.dilate(bin1, kernel1)  # 白色区域膨胀
    text = pytesseract.image_to_string(dilated, lang='chi_sim')  # 识别
    print('识别的结果为:%s' % text)


recognize_text(result)
