from ultralytics import YOLO

# TODO ---train l尺度模型
# 加载权重
model = YOLO("weights/once2.pt")

# 参数设置
# path = 'images'
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\psw"
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\nopsw"
# path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\nopsw_clear"
path = r"D:\CV\PSWDET_SOURCE\PSW_TESTSET\nopsw_conf6"
# path = r"D:\CV\PSWDET_SOURCE\240126DATA\long_videos\fortest\K56+366_CAM31_dvr_C1R27P475_15-54-39_15-58-59_0.mp4"

results = model.predict(source=path, conf=0.6, imgsz=1280, stream=True, device=[0], save=False, save_crop=False,
                        seg_filter=True, cls_filter=True, save_if_box_father=True)
# stream=True时必须添加这段代码
for r in results:
    pass

# 模型导出
#  model.export(format="engine", device=0, imgsz=640, half=True)

# 指定保存结果
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     img_name = os.path.basename(r.path)
#     print("保存...", img_name)
#     im.save(os.path.join('runs', img_name))
