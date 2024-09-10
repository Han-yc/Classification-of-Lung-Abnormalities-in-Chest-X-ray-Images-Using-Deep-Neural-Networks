# 测试某个模型在测试集上的最终准确率
# main函数上方有注释，注意看注释
import keras
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')
# 忽略AVX2 FMA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    # 需要根据需求进行更改的参数
    # ————————————————————————————————————————————————————————————————
    test_root = '../data/final/test'
    model_path = "../train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"
    # ————————————————————————————————————————————————————————————————

    # verbose：日志显示
    # verbose = 0 为不在标准输出流输出日志信息
    # verbose = 1 为输出进度条记录
    # verbose = 2 为每个epoch输出一行记录
    verbose = 1
    # 图片进行训练时，resize的大小
    IM_WIDTH = 224
    IM_HEIGHT = 224
    # 根据使用的机器的显存大小，调整batch_size的值，一般为2的幂次
    batch_size = 32

    # test data
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_datagen.flow_from_directory(
        test_root,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    model = load_model(model_path)
    result = model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['acc', keras.metrics.Precision(class_id=0, name='precision_macro'),
                                    keras.metrics.Recall(class_id=0, name='recall_macro')])
    model.evaluate_generator(test_generator, steps=test_generator.n / batch_size, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=verbose)
