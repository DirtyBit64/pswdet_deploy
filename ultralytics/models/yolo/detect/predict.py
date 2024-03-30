# Ultralytics YOLO 🚀, AGPL-3.0 license
import onnxruntime
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from ultralytics.models.yolo.detect.roadseg.seg_utils import seg_filter


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

    # 设置ONNX模型路径
    ONNX_MODEL_PATH = 'ultralytics/models/yolo/detect/roadseg/pidnet.onnx'
    # seg_model = load_pretrained(get_pred_model(name='s', num_classes=2),
    #                             pretrained='ultralytics/models/yolo/detect/roadseg/pidnet.pt').cuda()
    # 加载onnxruntime解释器
    seg_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH, None,
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
            # 采用道路分割掩码对nms后的预测框进行二次过滤
            if self.args.seg_filter and pred.shape[0] != 0:
                pred = seg_filter(pred, orig_img, self.seg_model, prop=0.8)

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
