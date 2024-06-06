# Ultralytics YOLO 🚀, AGPL-3.0 license
import onnxruntime
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from ultralytics.models.yolo.detect.roadseg.seg_utils import seg_filter
from ultralytics.models.yolo.detect.filtlercls.cls_utils import cls_filter
import numpy as np
import torch


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    ###########################################
    # 记录单次检测过程中box数目
    box_count = 0

    # 初始化加载分割模型
    SEG_ONNX_MODEL_PATH = 'ultralytics/models/yolo/detect/roadseg/pidnet.onnx'
    seg_model = onnxruntime.InferenceSession(SEG_ONNX_MODEL_PATH, None,
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 初始化加载分类模型
    CLS_ONNX_MODEL_PATH = 'ultralytics/models/yolo/detect/filtlercls/psw64a.onnx'
    cls_model = onnxruntime.InferenceSession(CLS_ONNX_MODEL_PATH, None,
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 记录上次检出的preds
    pre_boxes_np = []
    ###########################################

    def postprocess(self, preds, img, orig_imgs):
        """对单个预测结果进行后处理并返回results对象集合"""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            if is_list:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]

            ###########################################
            # 1.采用道路分割掩码对nms后的预测框进行二次过滤
            if self.args.seg_filter and pred.shape[0] != 0:
                pred = seg_filter(pred, orig_img, self.seg_model, prop=0.8)

            # 2.利用分类过滤模型对nms后的预测框进行二次过滤
            if self.args.cls_filter and pred.shape[0] != 0:
                pred = cls_filter(self.cls_model, pred, orig_img, proportion=0.2)

            # 3.时序iou去重
            if self.args.frame_iou > 0 and pred.shape[0] != 0:
                pred = self.frame_iou_filter(pred)

            # 检测框计数
            self.box_count += pred.shape[0]
            # 动态调整超参，有误报才保存
            if self.args.save_if_box_father:
                self.args.save_if_box = pred.shape[0] != 0

            ###########################################
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        ########################################
        print("当前总检测框数目: ", self.box_count)
        ########################################

        return results

    def frame_iou_filter(self, pred):
        """
            减少时序重复检测，用于视频流
                @pred 当前帧的检测结果
        """
        if self.args.frame_iou > 1:
            raise ValueError("frame_iou阈值必须在0-1的范围内")

        boxes_np = pred.cpu().numpy()

        # 拷贝副本
        temp_pre_boxes = self.pre_boxes_np.copy()
        temp_boxes_np = boxes_np.copy()

        # 更新pre_boxes_np
        self.pre_boxes_np = boxes_np

        # 基于mask过滤检测框
        r = 0
        for box_np in temp_boxes_np:  # (x1,y1,x2,y2,conf,cls)
            is_del = False
            for pre_box_np in temp_pre_boxes:
                # 类别不一样跳过
                if box_np[5] != pre_box_np[5]:
                    continue

                # 压缩成x1x2y1y2方便计算iou
                box1 = np.array([box_np[0], box_np[1], box_np[2], box_np[3]])
                box2 = np.array([pre_box_np[0], pre_box_np[1], pre_box_np[2], pre_box_np[3]])

                if calculate_iou(box1, box2) >= self.args.frame_iou:
                    boxes_np = np.delete(boxes_np, r, axis=0)
                    is_del = True
                    break

            if not is_del:
                r += 1

        # 返回过滤后的Tensor.cuda()类型结果
        return torch.tensor(boxes_np).cuda()


# from文心一言
# 计算两个框的iou,注意box要是numpy数组
def calculate_iou(box1, box2):
    # 获取两个边界框的坐标
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算两个边界框交集的左上和右下坐标
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    # 计算交集面积，如果交集不存在则面积为0
    if inter_x2 - inter_x1 < 0 or inter_y2 - inter_y1 < 0:
        intersection_area = 0
    else:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 计算两个边界框的并集面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou
