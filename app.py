import os
os.chdir('..\\darkflow') ##작업디렉토리 darkflow로하지 않으면 에러남...아나콘다만?
from time import time
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename, redirect
from main import predict


app =  Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    return render_template('index.html')

#파일 업로드 처리

@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    os.chdir('..\\Korean_Car_LPR') ##flask_upload_img에 저장하기위해 작업디렉토리변경
    if request.method == 'POST':
        f = request.files['file']
        #경로 바꾸기, 경로 설정 조사 필요 ~> 상대경로로
        fname = ('flask_upload_img\\'
                    + str(int(time())) + secure_filename(f.filename))

        print("fname:",fname)
        f.save(fname)
        # filename을 넘겨주고 오리지널 사진파일 이름, 관심영역(번호판) 사진파일 이름 받아옴
        # 이름은 중복되지않게 바꿔주므로 이렇게 받아와야함
        original_new_name, roi_new_name, predict_result = predict(fname)


    img_path = 'assets/img/result_img'


    path_original = img_path + '/'+original_new_name
    path_roi = img_path + '/'+roi_new_name
    print(path_original)
    print(path_roi)
    return render_template('result.html',predict_result_html = predict_result, original_file = path_original, roi_file = path_roi)

if __name__ == '__main__':
    # app.run(host='0,0,0,0',port='5001',debug=True)
    app.run(debug=True) #debug=True 의미 : 디버그 정보 보여줌 False가 기본설정