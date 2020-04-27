# Album_tensorflow

# 微信小程序——炫酷相册

## 功能介绍
  此项目为[Album](https://github.com/AsamuQ/Album)后台服务
  实现图片识别服务[Python+TensorFlow2.1]

## 目录结构
```c
├── wx_album                                    // 源码目录
│   ├── lan                                     // 控制器层
│   ├── border                                  // 训练记录
│   ├── dataset                                 // 数据集
│   |    ├── train                              // 训练集
│   |    ├── val                                // 验证集
│   ├── model                                   // 模型保存目录
│   |    ├── model_class.json                   // 类别记录
│   |    ├── xxx.h5                             // 训练模型
│   ├── result                                  // 识别结果
│   |    ├── classification.json                // 识别结果记录
│   ├── wxImage                                 // 识别图片目录 
│   ├── train.py                                // 训练代码 
│   ├── predict_server.py                       // 预测代码 
│   ├── TensorFlow环境配置.docx                  // 项目运行环境
```

## 备注
>项目包含一个训练集以及用于识别的图片，预测服务可直接运行
>环境配置说明已上传