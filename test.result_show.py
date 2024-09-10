import keras
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    test_root = '../data/final/test'
    models_    directory = "../train"  # 修改成你的模型文件夹路径
    model_files = glob.glob(models_directory + "/*.h5")  # 获取所有.h5文件

    verbose = 1
    IM_WIDTH = 224
    IM_HEIGHT = 224
    batch_size = 32

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_root,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    results = []

    for model_path in model_files:
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['acc', keras.metrics.Precision(class_id=0, name='precision_macro'),
                               keras.metrics.Recall(class_id=0, name='recall_macro')])
        result = model.evaluate(test_generator, steps=test_generator.n / batch_size, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=verbose)
        results.append((model_path, result))

    # 打印所有模型的结果
    for model_path, result in results:
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Loss: {result[0]}, Accuracy: {result[1]}, Precision: {result[2]}, Recall: {result[3]}")
