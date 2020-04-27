#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Contact :   mrlan2020@foxmail.com
@Modify Time      @Author          @Version    @Desciption
------------      -------          --------    -----------
2020/4/16 15:28   lanxiaokang      1.0         None
'''

# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import json
import os
from keras.callbacks import TensorBoard


batch_size = 8
epochs = 100
IMG_HEIGHT = 240
IMG_WIDTH = 180
num_classes=5
IMG_INPUT = 224

#使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = os.path.join(os.getcwd(),'dataset')


#设置训练集和验证集目录
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_bird_dir = os.path.join(train_dir, 'bird')
train_flower_dir = os.path.join(train_dir, 'flower')
train_instruments_dir = os.path.join(train_dir, 'instruments')
train_corn_dir = os.path.join(train_dir, 'corn')
train_mushroom_dir = os.path.join(train_dir, 'mushroom')

validation_bird_dir = os.path.join(validation_dir, 'bird')
validation_flower_dir = os.path.join(validation_dir, 'flower')
validation_instruments_dir = os.path.join(validation_dir, 'instruments')
validation_corn_dir = os.path.join(validation_dir, 'corn')
validation_mushroom_dir = os.path.join(validation_dir, 'mushroom')



num_bird_tr = len(os.listdir(train_bird_dir))
num_flower_tr = len(os.listdir(train_flower_dir))
num_instruments_tr = len(os.listdir(train_instruments_dir))
num_corn_tr = len(os.listdir(train_corn_dir))
num_mushroom_tr = len(os.listdir(train_mushroom_dir))

num_bird_val = len(os.listdir(validation_bird_dir))
num_flower_val = len(os.listdir(validation_flower_dir))
num_instruments_val = len(os.listdir(validation_instruments_dir))
num_corn_val = len(os.listdir(validation_corn_dir))
num_mushroom_val = len(os.listdir(validation_mushroom_dir))

#训练图片数
total_train = num_bird_tr+num_flower_tr+num_instruments_tr+num_corn_tr+num_mushroom_tr
total_val = num_bird_val + num_flower_val+num_instruments_val+num_corn_val+num_mushroom_val
print("Total training images:", total_train)
print("Total validation images:", total_val)


# 训练集
# 对训练图像应用了重新缩放，45度旋转，宽度偏移，高度偏移，水平翻转和缩放增强。
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    width_shift_range=0.1,
                    height_shift_range=0.1
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

# 验证集
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')




# 创建模型
model=tf.keras.applications.ResNet50(include_top=True, weights=None,classes=num_classes)
# 编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# 模型总结
model.summary()


# 模型保存格式定义
model_class_dir='./model/'
class_indices = train_data_gen.class_indices
class_json = {}
for eachClass in class_indices:
    class_json[str(class_indices[eachClass])] = eachClass
# 保存标签文件
with open(os.path.join(model_class_dir, "model_class.json"), "w+") as json_file:
    json.dump(class_json, json_file, indent=4, separators=(",", " : "),ensure_ascii=True)
    json_file.close()
print("JSON Mapping for the model classes saved to ", os.path.join(model_class_dir, "model_class.json"))


model_name = 'model_ex-{epoch:03d}_acc-{accuracy:03f}.h5'

trained_model_dir='./model/'
model_path = os.path.join(trained_model_dir, model_name)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
             filepath=model_path,
             monitor='accuracy',
            verbose=2,
            save_weights_only=True,
            save_best_only=True,
            mode='max',
            period=1)


def lr_schedule(epoch):
    # Learning Rate Schedule
    lr =1e-3
    total_epochs =epoch

    check_1 = int(total_epochs * 0.9)
    check_2 = int(total_epochs * 0.8)
    check_3 = int(total_epochs * 0.6)
    check_4 = int(total_epochs * 0.4)

    if epoch > check_1:
        lr *= 1e-4
    elif epoch > check_2:
        lr *= 1e-3
    elif epoch > check_3:
        lr *= 1e-2
    elif epoch > check_4:
        lr *= 1e-1

    return lr



lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,patience=5, min_lr=0.001)

num_train = len(train_data_gen.filenames)
num_test = len(val_data_gen.filenames)
print("num_train , num_test :")
print(num_train,num_test)


# 模型训练
# 使用fit_generator方法ImageDataGenerator来训练网络。
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(num_train / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(num_test / batch_size),
    callbacks=[checkpoint,lr_scheduler,TensorBoard(log_dir='border')])








