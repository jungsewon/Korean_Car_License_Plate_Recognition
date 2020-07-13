import os
# os.chdir('..\\darkflow\\darkflow\\net')  ##.. 하면 상위디렉토리로 가는듯!!!
# print("현재작업디렉토리: ",os.getcwd())


from darkflow.net.build import TFNet
import cv2
import numpy as np
import tensorflow as tf


##working directory darkflow 임 주의!! 일단 작업디렉토리 Korean_Car_LPR로 하고 해보자

options = {"model": "cfg\\tiny_car_LP_yolo.cfg", "load": -1, "threshold": 0.5}


tf.compat.v1.enable_eager_execution() ## 프로그램 시작할 때 한번만 실행해야 함

tfnet = TFNet(options)
def yolo_npd(image):
    try:

        # os.chdir('..\\darkflow\\darkflow\\net\\build.py')  ##.. 하면 상위디렉토리로 가는듯!!!
        from darkflow.net.build import TFNet

        ##한글이름 사진파일도 인식할수 있게
        stream = open(image, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img_ori = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        result = tfnet.return_predict(img_ori)
        # img_ori[result[0]['topleft']['x']]

        #result 값 예시 - 딕셔너리 형태로
        # [{'label': 'plate', 'confidence': 0.9012289, 'topleft': {'x': 478, 'y': 899}, 'bottomright': {'x': 748, 'y': 972}}]

        tl_x = result[0]['topleft']['x']
        tl_y = result[0]['topleft']['y']
        br_x = result[0]['bottomright']['x']
        br_y = result[0]['bottomright']['y']
        # print(result[0]['topleft']['x'])
        # print(result[0]['topleft']['y'])
        # print(result[0]['bottomright']['x'])
        # print(result[0]['bottomright']['y'])
        padding = 5 #이미지 전처리 없는 현재상태 (5) 에서
                    # 패딩값을 크게 주면 정답률 떨어짐. roi(관심영역) 이미지 전처리 반드시 해야함!!!
        roi = img_ori[tl_y-padding:br_y+padding, tl_x-padding:br_x+padding]

        # cv2.imshow('roi',roi)
        # cv2.waitKey(0)

        currentpath = os.getcwd()
        print(currentpath)

        # os.chdir('C:\\Users\\tsin2\\Desktop\\car_number_recognition\\Korean-License-Plate-Recognition-master')

        return roi, True
        # numPredict(roi)








    except Exception as e:  # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
        print("오류발생")
        print(e)
        return image, False




#
#
# imgcv = cv2.imread("C:\\data\\testset\\57.jpg")
# result = tfnet.return_predict(imgcv)
# print(result)





# from darkflow.net.build import TFNet
# import numpy as np
# import cv2
# import time
# import pprint as pp
#
# options = {
#             "model": "C:\\Users\\tsin2\\darkflow\\cfg\\tiny_car_LP_yolo.cfg",
#             "load": -1,
#             "threshold": 0.5
#         }
#
# tfnet2 = TFNet(options)
#
# tfnet2.load_from_ckpt()
#
# def boxing(original_img, predictions):
#     newImage = np.copy(original_img)
#
#     for result in predictions:
#         top_x = result['topleft']['x']
#         top_y = result['topleft']['y']
#
#         btm_x = result['bottomright']['x']
#         btm_y = result['bottomright']['y']
#
#         confidence = result['confidence']
#         label = result['label'] + " " + str(round(confidence, 3))
#
#         if confidence > 0.06:
#             newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
#             newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
#
#     return newImage
#
# original_img = cv2.imread("C:\\data\\testset\\57.jpg")
# original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
# result = tfnet2.return_predict(original_img)
#
# new_frame = boxing(original_img, result)
#
# cv2.imwrite('output.jpg', new_frame)