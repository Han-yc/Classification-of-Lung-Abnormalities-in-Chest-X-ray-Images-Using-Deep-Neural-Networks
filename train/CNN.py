# -*- coding: UTF-8 -*-
import os
import time
import warnings
import tensorflow as tf
# 在pycharm中调用 ‘tensorflow.keras’ 库时会有红色波浪线，不用担心是正常现象。
# 加载keras.api下的keras并返回一个变量或实例，用以调用keras下的方法。而这一过程必须在运行时才能发生。
# pycharm一类的解释器，识别不了这种调用方式，是导致红线和无法自动补全的原因，但是代码是没问题的，不用在意就行。
# 还想了解更加具体的原因，可以看下keras的发展历史，以及和Tensorflow框架的关系
from tensorflow.keras import metrics, losses, optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.layers import Dense, GlobalAveragePooling2D, \
    Dropout, multiply, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')
# 忽略AVX2 FMA的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    # Tensorflow 按需分配显存
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if not os.path.exists('./csv/'):
        os.mkdir('./csv/')

    # 以下是根据不同任务需求，需要进行手动修改的参数：
    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 选择要训练的卷积神经网络模型
    # 0 : VGG16
    # 1 : VGG19
    # 2 : ResNet50
    # 3 : ResNet101
    # 4 : ResNet152
    # 5 : ResNet50_V2
    # 6 : ResNet101_V2
    # 7 : ResNet152_V2
    # 8 : MobileNet_V1
    # 9 : MobileNet_V2
    # 10: MobileNet_V3_Small
    # 11: MobileNet_V3_Large
    # 12: DenseNet121
    # 13: DenseNet169
    # 14: DenseNet201
    # 15 : EfficientNet_B0
    # 16 : EfficientNet_B1
    # 17 : EfficientNet_B2
    # 18 : EfficientNet_B3
    # 19 : EfficientNet_B4
    # 20 : EfficientNet_B5
    # 21 : EfficientNet_B6
    # 22 : EfficientNet_B7
    # 23 : EfficientNetV2S
    # 24 : EfficientNetV2M
    # 25 : EfficientNetV2L
    # 26 : Inception_V3 (GoogLeNet)
    # 27 : Xception
    # 28 : NASNetMobile

    CNN_name = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                'ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'MobileNet_V1', 'MobileNet_V2',
                'MobileNet_V3_Small', 'MobileNet_V3_Large', 'DenseNet121', 'DenseNet169', 'DenseNet201',
                'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'EfficientNet_B3', 'EfficientNet_B4',
                'EfficientNet_B5', 'EfficientNet_B6', 'EfficientNet_B7', 'EfficientNetV2S', 'EfficientNetV2M',
                'EfficientNetV2L', 'Inception_V3', 'Xception', 'NASNetMobile', 'NASNetLarge'
                ]
    CNN_serial_number = 3

    # NB_CLASS: 分类的类别数量
    NB_CLASS = 3

    my_momentum = 0.9

    # 预训练模型路径
    pre_model_path = '../pretrained/'

    # 是否进行数据增强，0：不扩充数据，1：扩充数据
    augmentation_name = ['No_Aug', 'Aug']
    augmentation = 0

    # 是否添加注意力机制层，0：不增加，1：增加
    attention_name = ['No_Att', 'Att']
    attention = 1

    batch_size = 8

    # 总共的训练轮次
    EPOCH = 30

    # 多线程的数量，不支持的话换成1
    workers = 4

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 图片进行训练时，resize的大小
    IM_WIDTH = 224
    IM_HEIGHT = 224
    # 数据集路径
    train_root = '../data/final/train'
    validation_root = '../data/final/valid'
    train_root_augmentation = '../data/final/train_Augmentation'
    validation_root_augmentation = '../data/final/valid_Augmentation'

    # verbose：日志显示
    # verbose = 0 为不在标准输出流输出日志信息
    # verbose = 1 为输出进度条记录
    # verbose = 2 为每个epoch输出一行记录
    # 注意： 默认为 1
    verbose = 1

    # ——————————————————————————————————————————————————————————
    # 交叉熵损失函数，常用，默认使用这个即可
    loss = losses.categorical_crossentropy

    # Adam 优化器。常用的自适应优化器，参数如下：
    # ——————————————————————————————
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 数据生成器, 处理数据
    My_ImageDataGenerator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1. / 255
    )
    # 如果augmentation为0，使用train_root和validation_root,
    # 如果为1，使用train_root_augmentation和validation_root_augmentation
    if augmentation == 0:
        if not os.path.exists(train_root):
            print('请检查:' + train_root + ' 路径中训练集是否存在')
            print('请检查:' + train_root + '  路径中验证集是否存在')
        train_generator = My_ImageDataGenerator.flow_from_directory(
            train_root,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
            shuffle=True
        )
        vaild_generator = My_ImageDataGenerator.flow_from_directory(
            validation_root,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
        )
    elif augmentation == 1:
        train_generator = My_ImageDataGenerator.flow_from_directory(
            train_root_augmentation,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
            shuffle=True
        )
        vaild_generator = My_ImageDataGenerator.flow_from_directory(
            validation_root_augmentation,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
        )
    else:
        print("请检查augmentation参数的值是否合理")

    # ————————————————————————————————————————————————————————————————————————
    # 按照不同的序号，构建不同的模型
    if CNN_serial_number == 0:
        base_model = applications.VGG16(weights=pre_model_path + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        include_top=False)
    elif CNN_serial_number == 1:
        base_model = applications.VGG19(weights=pre_model_path + 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        include_top=False)
    elif CNN_serial_number == 2:
        base_model = applications.ResNet50(
            weights=pre_model_path + 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            layers=tf.keras.layers, include_top=False)
    elif CNN_serial_number == 3:
        base_model = applications.ResNet101(
            weights=pre_model_path + 'resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5',
            layers=tf.keras.layers, include_top=False)
    elif CNN_serial_number == 4:
        base_model = applications.ResNet152(
            weights=pre_model_path + 'resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5',
            layers=tf.keras.layers, include_top=False)
    elif CNN_serial_number == 5:
        base_model = applications.ResNet50V2(
            weights=pre_model_path + 'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 6:
        base_model = applications.ResNet101V2(
            weights=pre_model_path + 'resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 7:
        base_model = applications.ResNet152V2(
            weights=pre_model_path + 'resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 8:
        base_model = applications.MobileNet(weights=pre_model_path + 'mobilenet_1_0_224_tf_no_top.h5',
                                            include_top=False)
    elif CNN_serial_number == 9:
        base_model = applications.MobileNetV2(
            weights=pre_model_path + 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
            include_top=False)
    elif CNN_serial_number == 10:
        base_model = applications.MobileNetV3Small(
            weights=pre_model_path + 'weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5',
            include_top=False)
    elif CNN_serial_number == 11:
        base_model = applications.MobileNetV3Large(
            weights=pre_model_path + 'weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5',
            include_top=False)
    elif CNN_serial_number == 12:
        base_model = applications.DenseNet121(
            weights=pre_model_path + 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 13:
        base_model = applications.DenseNet169(
            weights=pre_model_path + 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 14:
        base_model = applications.DenseNet201(
            weights=pre_model_path + 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 15:
        base_model = applications.EfficientNetB0(
            weights=pre_model_path + 'efficientnetb0_notop.h5', include_top=False)
    elif CNN_serial_number == 16:
        base_model = applications.EfficientNetB1(
            weights=pre_model_path + 'efficientnetb1_notop.h5', include_top=False)
    elif CNN_serial_number == 17:
        base_model = applications.EfficientNetB2(
            weights=pre_model_path + 'efficientnetb2_notop.h5', include_top=False)
    elif CNN_serial_number == 18:
        base_model = applications.EfficientNetB3(
            weights=pre_model_path + 'efficientnetb3_notop.h5', include_top=False)
    elif CNN_serial_number == 19:
        base_model = applications.EfficientNetB4(
            weights=pre_model_path + 'efficientnetb4_notop.h5', include_top=False)
    elif CNN_serial_number == 20:
        base_model = applications.EfficientNetB5(
            weights=pre_model_path + 'efficientnetb5_notop.h5', include_top=False)
    elif CNN_serial_number == 21:
        base_model = applications.EfficientNetB6(
            weights=pre_model_path + 'efficientnetb6_notop.h5', include_top=False)
    elif CNN_serial_number == 22:
        base_model = applications.EfficientNetB7(
            weights=pre_model_path + 'efficientnetb7_notop.h5', include_top=False)
    elif CNN_serial_number == 23:
        base_model = applications.EfficientNetV2S(
            weights=pre_model_path + 'efficientnetv2-s_notop.h5', include_top=False)
    elif CNN_serial_number == 24:
        base_model = applications.EfficientNetV2M(
            weights=pre_model_path + 'efficientnetv2-m_notop.h5', include_top=False)
    elif CNN_serial_number == 25:
        base_model = applications.EfficientNetV2L(
            weights=pre_model_path + 'efficientnetv2-l_notop.h5', include_top=False)
    elif CNN_serial_number == 26:
        base_model = applications.InceptionV3(
            weights=pre_model_path + 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 27:
        base_model = applications.Xception(
            weights=pre_model_path + 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number == 28:
        base_model = applications.NASNetMobile(
            weights=pre_model_path + 'nasnet_mobile_no_top.h5', include_top=False)
    elif CNN_serial_number == 29:
        base_model = applications.NASNetLarge(
            weights=pre_model_path + 'nasnet_large_no_top.h5', include_top=False)
    else:
        print("请检查CNN_serial_number参数的值是否合理")

    x = base_model.output

    if attention == 1:
        in_put = x
        x = GlobalAveragePooling2D()(in_put)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        x = multiply([in_put, x])
    elif attention == 0:
        pass
    else:
        print("请检查attention参数的值是否合理")

    # 添加全局平均池化层
    x = GlobalAveragePooling2D(name='flag')(x)
    # 添加一个全连接层
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # 添加一个分类器，假设我们有NB_CLASS个类
    predictions = Dense(NB_CLASS, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    add_flag = False

    # 迁移学习,冻结预训练模型的权重，只训练自己加的层和模型中的BN层
    for layer in model.layers:
        if 'flag' == layer.name:
            add_flag = True
        layer.trainable = add_flag
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
            layer.momentum = my_momentum

    # 打印模型最终结构
    model.summary()

    # 编译模型
    # model.compile(optimizer=optimizer, loss=loss,
    # z             metrics=['acc', metrics.Precision(), metrics.Recall()])

    # 使用下面这个
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['acc', metrics.Precision(class_id=0, name='precision_macro'),
                           metrics.Recall(class_id=0, name='recall_macro')])

    # 日志存放路径
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    log_dir = './logs/' + CNN_name[CNN_serial_number] + "_" + attention_name[attention] + "_" + \
              augmentation_name[augmentation] + "_" + otherStyleTime
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch", )
    cvs_path = './csv/' + CNN_name[CNN_serial_number] + "_" + attention_name[attention] + "_" + \
               augmentation_name[augmentation] + "_" + otherStyleTime + '.csv'
    csv_logger = CSVLogger(cvs_path, append=True)

    # 最优模型的保存路径
    filePath = './' + CNN_name[CNN_serial_number] + "_" + attention_name[attention] + "_" + \
               augmentation_name[augmentation] + "_" + otherStyleTime + '_best.h5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', save_freq="epoch")
    callbacks_list = [checkpoint, tbCallBack, csv_logger]

    # 在数据集上训练
    history = model.fit(train_generator, validation_data=vaild_generator,
                        epochs=EPOCH,
                        steps_per_epoch=train_generator.n / batch_size,
                        validation_steps=vaild_generator.n / batch_size, verbose=verbose,
                        callbacks=callbacks_list, workers=workers)
    print('模型训练完成！')
