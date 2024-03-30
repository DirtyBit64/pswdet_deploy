# ------------------------------------------------------------------------------
# Written by ljx ---- 2024.2.1
# ------------------------------------------------------------------------------
import math
import numpy as np
import torch
from PIL import Image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def onnx_infer(img, session):
    # 模型的输入输出名，必须和onnx的输入输出名相同，可以通过netron查看，如何查看参考下文
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # run方法用于模型推理，run(输出张量名列表，输入值字典)
    return session.run([output_name], {input_name: img})


def seg_filter(boxes, orig_img, seg_model, prop):
    """
        采用道路语义分割掩码对nms后的预测框进行过滤
            @orig_img 原图
            @boxes orig_img 经过nms后的检测结果
            @weight_path 分割权重类路径
    """
    # 先将原图送入分割模型获得分割掩码
    img = input_transform(orig_img)
    img = img.transpose((2, 0, 1)).copy()
    img = np.expand_dims(img, axis=0)  # 3维转4维
    # 推理
    mask = onnx_infer(img, seg_model)[0]

    # For test
    # check_seg_display(mask, img, orig_img)

    # 将检测框Tensor变量转为Numpy数组
    boxes_np = boxes.cpu().numpy()

    # 基于mask过滤检测框
    r = 0
    for box_np in boxes_np:
        if not is_road(mask=mask, box=box_np, proportion=prop):
            boxes_np = np.delete(boxes_np, r, axis=0)
            continue
        r += 1

    # 返回过滤后的Tensor.cuda()类型结果
    return torch.tensor(boxes_np).cuda()


def is_road(mask, box, proportion):
    """
        判断box范围内的mask占比是否超过proportion
            @mask 分割掩码
            @box 检测框数据
            @proportion 分割过滤器占比因子
    """
    if proportion > 1 or proportion < 0:
        raise ValueError("seg_filter占比因子必须在0-1的范围内")

    # 获取像素坐标
    x1 = math.ceil(box[0])
    x2 = math.ceil(box[2])
    y1 = math.ceil(box[1])
    y2 = math.ceil(box[3])

    return np.sum(mask[:, y1:y2, x1:x2] == 1) / ((y2 - y1) * (x2 - x1)) > proportion


def check_seg_display(mask, img, orig_img):
    """
        通过输出可视化检查嵌入的分割模块是否正确分割
            @mask 分割掩码
            @img 特征图
            @orig_img 原图
    """
    mask = torch.tensor(mask).squeeze(0).cpu().numpy()
    color_map = [(128, 64, 128),
                 (244, 35, 232)]
    img = np.squeeze(img, axis=0)  # 4维转3维
    img = img.transpose([1, 2, 0])
    sv_img = np.zeros_like(img).astype(np.uint8)
    for i, color in enumerate(color_map):
        for j in range(3):
            sv_img[:, :, j][mask == i] = color_map[i][j]
    sv_img = Image.fromarray(sv_img)
    sv_img.save("bug.jpg")
    # Overlay Vis
    img = Image.fromarray(orig_img).convert('RGB')
    sv_img = sv_img.convert('RGB')
    overlay = Image.blend(sv_img, img, 0.6)
    overlay.save("bug2.jpg")
