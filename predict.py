#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   predict.py    
@Contact :   mrlan2020@foxmail.com
@Modify Time      @Author          @Version    @Desciption
------------      -------          --------    -----------
2020/4/21 13:52   lanxiaokang      1.0         None
'''

 
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import  numpy as np
import os
import json
CLASS_INDEX = None
input_image_size=224
class_num=5

model_jsonPath='./model/model_class.json'


def preprocess_input(x):
    x *= (1./255)
    return x


def decode_predictions(preds, top=5, model_json=""):

    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)
    return results


def create_model():
    base_model=ResNet50(include_top=True, weights=None,classes=class_num)
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    return model

model=create_model()
# 编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('./model/model_ex-091_acc-0.997391.h5')

# 预测结果保存为json文件
def save_to_json(path,json_data):
    fr = open(path)
    model = json.load(fr)
    fr.close()

    for i in json_data:
        model[i] = json_data[i]

    jsObj = json.dumps(model)

    with open(path, "w") as fw:
        fw.write(jsObj)

#图像预测
result_file = os.path.join(os.getcwd(),"wxImage")
for root,dirs,files in os.walk(result_file):
    for item in files:
        image_path = root+"\\"+item
        image_to_predict = image.load_img(image_path)
        # 图像预处理（此ResNet网络要求输入shape(224,224,3)）
        image_to_predict = image_to_predict.convert('RGB')  # 转化为灰度图
        x1 = image.img_to_array(image_to_predict)
        x1.resize((input_image_size, input_image_size, 3))
        x1 = np.expand_dims(x1, axis=0)  # 增加一个维度
        np.asarray(image_to_predict, dtype=np.float64)  # float64=double
        x1 = preprocess_input(x1)
        prediction = model.predict(x1)

        prediction_results = []
        prediction_probabilities = []
        try:
            predictiondata = decode_predictions(prediction, top=int(class_num), model_json=model_jsonPath)

            for result in predictiondata:
                prediction_results.append(str(result[0]))
                prediction_probabilities.append(result[1] * 100)
        except:
            raise ValueError("An error occured! Try again.")

        # 将结果保存进json文件
        json_file = os.path.join(os.getcwd(), "result", "classification.json")
        json_data = {prediction_results[0]: prediction_probabilities[0]}
        print("预测结果：", prediction_results[0], prediction_probabilities[0])
        save_to_json(json_file,json_data)


#单张图的所有预测结果
# for eachPrediction, eachProbability in zip(prediction_results, prediction_probabilities):
#     print(eachPrediction, " : ", eachProbability)