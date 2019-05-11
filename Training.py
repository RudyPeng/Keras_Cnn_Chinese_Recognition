from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model

data_dir = './data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# dimensions of our images.
# 统一格式64*64
# 字符集大小为12
img_width, img_height = 64, 64
charset_size = 17
nb_validation_samples = 800
# 每次epoch验证的数量
nb_samples_per_epoch = 2000
# 每次epoch训练的数据数量
nb_nb_epoch = 16


def train(model):
    # 图片生成器，批量生产数据防止过拟合
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # 重缩放因子 RGB系数太高，无法处理，所以用1/255进行缩放处理
        rotation_range=0.1,
        # 整数，随机旋转的度数范围
        # 随机垂直或水平平移图片的范围
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # 测试数据只进行重缩放

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        # 图片尺寸
        batch_size=3,
        # 一批数据的大小，用一半甚至更少的数据训练出来的梯度与用全部数据训练出来的梯度基本是一样的
        color_mode="grayscale",
        # 灰度图像
        class_mode='categorical')
        # 2d one-hot编码标签
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=3,
        color_mode="grayscale",
        class_mode='categorical')
    # 测试数据，一切同上
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # 编译，损失函数，优化器，列表
    model.fit_generator(train_generator,
                        samples_per_epoch=nb_samples_per_epoch,
                        nb_epoch=nb_nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_validation_samples)
    # 用python生成器逐批生成数据，按批次训练，提高效率


def build_model(include_top=True, input_shape=(64, 64, 1), classes=charset_size):
    img_input = Input(shape=input_shape)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    # 第一层 32个卷积核 3*3
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    # 第二层
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # 第一个最大池化层
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    # 第三层
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    # 第四层
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # 第二个最大池化层
    if include_top:
        x = Flatten(name='flatten')(x)
        # 压平层，从卷积层到全连接层的过度
        x = Dropout(0.2)(x)
        # 防止过拟合，丢掉20%
        x = Dense(1024, activation='relu', name='fc2')(x)
        # 全连接层
        x = Dense(classes, activation='softmax', name='predictions')(x)
        # 全连接层+softmax

    model = Model(img_input, x, name='model')
    # Model模型比Sequential模型效果更好，序贯模型是Model模型的简略版，书写结构不同
    return model


model = build_model()
train(model)
model.save("model2.h5")