# ------------------------------------------------------------------------------
# Written by ljx ---- 2024.4.9
# ------------------------------------------------------------------------------
import math
import os.path

import cv2
import numpy as np
import torch


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def cls_filter(model, boxes, orig_img, proportion):
    """
        分类过滤器，排除裂缝、阴影、箭头等误报
            @boxes 检测框
            @orig_img 原图
            @proportion 分割过滤器对比因子
    """
    if proportion > 1 or proportion < 0:
        raise ValueError("cls_filter对比因子必须在0-1的范围内")

    # 将检测框Tensor变量转为Numpy数组
    boxes_np = boxes.cpu().numpy()
    for box_np in boxes_np:

        # 将YOLO格式坐标转换为边界框左上角和右下角的坐标
        # 获取像素坐标
        x1 = math.ceil(box_np[0])
        y1 = math.ceil(box_np[1])
        x2 = math.ceil(box_np[2])
        y2 = math.ceil(box_np[3])

        # 裁剪原图
        image = orig_img[y1:y2, x1:x2, :]
        image = cv2.resize(image, (64, 64))  # 与导出参数对齐
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = input_transform(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)  # 3维转4维

        # 将检测框图片送入分类模型
        cls_res = softmax(onnx_infer(image, model)[0])[0]

        # 删除不符合的box_np
        r = 0
        for _ in boxes_np:
            maxsy = np.argmax(cls_res)
            if maxsy != 1 and (cls_res[maxsy] > proportion):
                boxes_np = np.delete(boxes_np, r, axis=0)
                continue
            ################################ FOR DEBUG
            # path = os.path.join('bugorig', str(cls_res[1]) + '.jpg')
            # cv2.imwrite(path, orig_img)
            ################################
            r += 1

    return torch.tensor(boxes_np).cuda()


def onnx_infer(img, session):
    # 模型的输入输出名，必须和onnx的输入输出名相同，可以通过netron查看，如何查看参考下文
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # run方法用于模型推理，run(输出张量名列表，输入值字典)
    return session.run([output_name], {input_name: img})


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 防止指数溢出
    return e_x / np.sum(e_x)
