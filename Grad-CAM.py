# 激活热力图可视化
# main函数中model_path指向需要可视化的模型，可视化的图片在pic文件夹中
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
from tensorflow.keras.applications.resnet50 import preprocess_input


def save_and_display_gradcam(img_path, heatmap, cam_path="./result/cam.jpg", alpha=0.5):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def cam_pic(img_path, model_path, save_path, last_conv_layer_name):
    img_size = (224, 224)
    model = keras.models.load_model(model_path)
    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    # Remove last layer's softmax
    model.layers[-1].activation = None
    # Print what the top predicted class is
    preds = model(img_array)
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap, save_path)
    print('已保存到'+save_path)


def cam_Folder(folder_path, model_path, save_FolderPath, last_conv_layer_name):
    for category in os.listdir(folder_path):
        path = folder_path + '/' + category
        save_path = save_FolderPath + '/' + category
        cam_pic(path, model_path, save_path, last_conv_layer_name)
        keras.backend.clear_session()


if __name__ == '__main__':
    model_path = "../train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"
    # 最后一个卷积层的名称
    last_conv_layer_name = "conv5_block3_out"
    # 可视化一张图片
    # img_path = './pic/Grape___Esca_(Black_Measles)1.jpg'
    # save_path = "./result/cam.jpg"
    # cam_pic(img_path, model_path, save_path, last_conv_layer_name)

    # 可视化一个文件里所有的图片
    folder_path = './pic'
    save_FolderPath = './Grad_CAM_result'
    cam_Folder(folder_path, model_path, save_FolderPath, last_conv_layer_name)
