# 绘制混淆矩阵
# main函数上方有注释，注意看注释
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

warnings.filterwarnings('ignore')
# 忽略AVX2 FMA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test(picname, model):
    keras.backend.clear_session()
    img_path = picname
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    predict = model.predict(test_image)
    preds = np.argmax(predict, axis=1)[0]
    return preds


if __name__ == '__main__':
    # 需要根据需求进行更改的参数
    # ————————————————————————————————————————————————————————————————
    test_root = '../data/final/test'
    model_path = "../train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"
    num_class = 3
    # 模型不同类别的名称，默认已0，1，2，3......代替不同的类别名称，想要自定义的话，可以自己传列表进来。
    display_labels = range(0, num_class)
    # 根据使用的机器的显存大小，调整batch_size的值，一般为2的幂次
    batch_size = 32
    # 是否要在混淆矩阵每个单元格上显示具体数值
    show_figure = False
    # 是否要对结果进行归一化
    normalization = True
    # ————————————————————————————————————————————————————————————————

    IM_WIDTH = 224
    IM_HEIGHT = 224
    # test data
    test_datagen = image.ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_datagen.flow_from_directory(
        test_root,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        shuffle=False
    )

    model = load_model(model_path)
    result = model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['acc', keras.metrics.Precision(), keras.metrics.Recall()])
    prediction = model.predict_generator(test_generator, verbose=1)
    y_pred = np.argmax(prediction, axis=1)
    y_true = test_generator.classes
    if normalization:
        # 归一化
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    else:
        # 不进行归一化
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    for i in range(0, len(cm)):
        print(cm[i])
        print("____________________________________________")

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(0, num_class))
    disp.plot(
        include_values = show_figure,  # 混淆矩阵每个单元格上显示具体数值
        cmap="Blues",  # 混淆矩阵常用颜色空间
        ax=None,
        xticks_rotation="horizontal"
    )
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    save_dir = './result/ConfusionMatrix_' + otherStyleTime + '_.png'
    # plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    plt.show()
