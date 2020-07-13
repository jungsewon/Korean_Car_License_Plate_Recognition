
from time import time

import numpy as np
# import os,sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import LPRNet
from loader import resize_and_normailze


classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "로",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]



def numPredict(image):
    # tf.compat.v1.enable_eager_execution() ## 프로그램 시작할 때 한번만 실행해야 함
    net = LPRNet(len(classnames) + 1)
    net.load_weights('saved_models/weights_best.pb')
    img = image
    # img = cv2.imread(image)
    x = np.expand_dims(resize_and_normailze(img), axis=0)
    predict_time1 = time()
    result = net.predict(x, classnames)
    # print(result)
    predict_time2 = time() - predict_time1
    print("번호판 판별시간: ", predict_time2)

    return result
