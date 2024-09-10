# 模型即使不搞什么网站，或者GUI程序，之间写一个脚本，也能在命令行里实现对指定的图片的识别
# 这段程序就是该脚本，使用main函数中model_path指向的模型文件，去检测base_path指向的文件夹里的图片
# model_path和base_path按自己的要求填写
import os
import warnings
from tensorflow import keras
import openpyxl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

warnings.filterwarnings('ignore')
# 忽略AVX2 FMA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test(picname, model, confidence=0):
    keras.backend.clear_session()
    img_path = picname
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    predict = model.predict(test_image)
    preds = np.argmax(predict, axis=1)[0]
    print("最大值概率为：" + str(predict[0][preds]))
    if predict[0][preds] < confidence:
        finalResult = '无关图片'
    else:
        myExcel = openpyxl.load_workbook('../data/message.xlsx')  # 获取表格文件
        mySheet = myExcel['Sheet1']  # 获取指定的sheet
        finalResult = (mySheet.cell(row=preds + 2, column=2)).value

    return finalResult


if __name__ == '__main__':
    base_path = './pic'  # 识别pic文件夹下的图片
    # 置信度
    confidence = 0
    xg = '/'
    model_path = "../train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    result = ['']

    for category in os.listdir(base_path):
        path = base_path + xg + category
        print("图片地址为： " + path)
        print("识别结果为： " + test(path, model, confidence))
        print("____________________________")
        keras.backend.clear_session()
