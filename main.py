import os
os.chdir('..\\darkflow')

from time import time
from num_predict import numPredict
from yolo_num_plate_detect import yolo_npd
import cv2
import numpy as np


### 직선검출위한 허프변환 ###
def hough(img):
    try:
        img_original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        # threshold값을 너무 높게 주면 선이 하나도 안 그어져 에러 생김
        # 에러 처리 코드 보완이 필요!!

        angle_sum = 0

        theta = 0
        for i in range(len(lines)):
            # print(lines[i])
            for j in range(len(lines[i])):
                angle_sum += np.degrees(lines[i][j][1])
                # print(np.degrees(lines[i][j][1]))
        avg_angle = 90 - (angle_sum / len(lines))

        print("평균각도: ", avg_angle)

        height, width, c = img.shape

        plate_cx = int(width / 2)
        plate_cy = int(height / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=-avg_angle, scale=1.0)
        img_rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(width, height))
        return img_rotated
    except:  ## 에러나면 들어온값 그대로 반환 ##
        return img_original
#####################################################################################################################################################################
#####################################################################################################################################################################

def predict(fname):

    # os.chdir('..\\darkflow') ##.. 하면 상위디렉토리로 가는듯!!!

    detect_time1 = time()
    roi, flag = yolo_npd(fname)

    if flag:

        detect_time2 = time() - detect_time1
        print("번호판 검출시간: ", detect_time2)
        # 허프 변환
        roi_rotated = hough(roi)
        # 딥러닝 LPRNet 이용한 번호판 인식
        predict_result = numPredict(roi_rotated)  # predict_result 는 리스트
        print(predict_result)

        stream = open(fname, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        ori_img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        ##경로 바꾸고 상대경로로 해야함 ~> 코드개선필요

        original_new_name = "ori" + str(int(time())) + ".jpg"
        roi_new_name = "roi" + str(int(time())) + ".jpg"
        path = 'static\\assets\\img\\result_img'


        print(os.path.join(path, original_new_name))
        cv2.imwrite(os.path.join(path, original_new_name), ori_img)
        cv2.imwrite(os.path.join(path, roi_new_name), roi_rotated)

        return original_new_name, roi_new_name, predict_result[0]
