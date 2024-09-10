import os
import numpy as np
from tensorflow import keras
from PIL import Image
from keras.models import load_model
from keras.utils import image_utils


def prepicture(picname):
    img = Image.open('./media/pic/' + picname)
    new_img = img.resize((100, 100), Image.BILINEAR)
    new_img.save(os.path.join('./media/pic/', os.path.basename(picname)))


def read_image2(filename):
    img = Image.open('./media/pic/' + filename).convert('RGB')
    return np.array(img)


def MyPrediction(picname):
    keras.backend.clear_session()

    # Django网站使用的模型的地址
    model_path = "./train/ResNet50_Att_No_Aug_2024-07-10-22-49-29_best.h5"

    img_path = './media/pic/' + picname
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    res_m = model(test_image, training=False)
    res = np.array(res_m)
    real_confidence = res.max()
    preds = np.argmax(res)
    return preds, real_confidence
