# 测试某个模型在测试集上的最终准确率
# main函数上方有注释，注意看注释
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras_preprocessing.image import load_img, img_to_array


# 卷积网络可视化
def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    num = f1.shape[-1]
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), i + 1)
        plt.imshow(f1[0, :, :, i] * 255, cmap='gray')
        plt.axis('off')
    save_path = "./result/visualization_layer_" + str(num_layer) + ".png"
    plt.savefig(save_path)
    # plt.show()


def main(model_path, num_layer, image):
    model = load_model(model_path)
    images = cv2.imread(image)
    # Turn the image into an array.
    # image_arr = process_image(image, (224, 224, 3))  # 根据载入的训练好的模型的配置，将图像统一尺寸
    # image_arr = np.expand_dims(image_arr, axis=0)
    img_size = (224, 224)
    image = load_img(image, target_size=img_size)
    x = img_to_array(image)
    image_arr = np.expand_dims(x, axis=0)
    image_arr = image_arr / 255
    # image_arr = preprocess_input(image_arr)
    # 设置可视化的层
    visual(model, image_arr, num_layer=num_layer)


if __name__ == '__main__':
    # 可视化的图片的地址
    image = './pic/Pneumonia_2252.jpg'
    # 可视化的模型的地址
    model_path = "../train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"
    # 可视化模型的第几层
    for i in range(4, 10):
        main(model_path, i, image)


