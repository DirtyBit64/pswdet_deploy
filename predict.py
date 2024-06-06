from ultralytics import YOLO

# 加载模型
model = YOLO("weights/once2.pt")

# 路径设置
# path = 'images'
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\psw"
# path = r"D:\CV\行人摩托车标注\20240415_ljx\images"
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\nopsw"
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\nopsw_clear"
# path = r"D:\CV\PSWDET_SOURCE\部署\重复样本"
# path = r"D:\CV\PSWDET_SOURCE\240126DATA\long_videos\fortest\K56+366_CAM31_dvr_C1R27P475_15-54-39_15-58-59_0.mp4"
path = r"D:\CV\PSWDET_SOURCE\部署\20240524视频\01010004906000000.mp4"

results = model.predict(source=path, conf=0.63, imgsz=(736, 1280), stream=True, device=[0], save=False, save_crop=False,
                        vid_stride=25, seg_filter=True, cls_filter=True, save_if_box_father=True, frame_iou=0.7)
# stream=True时必须添加这段代码
for r in results:
    pass

# 指定保存结果
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     img_name = os.path.basename(r.path)
#     print("保存...", img_name)
#     im.save(os.path.join('runs', img_name))
