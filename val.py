from ultralytics import YOLO

# 加载模型
model = YOLO("weights/yolofake/weights/once2.pt")


# Validate the model
dataset_path = r"ultralytics/cfg/datasets/PSW_TESTSET.yaml"
metrics = model.val(data=dataset_path, imgsz=960, batch=4, conf=0.3, device=0)

print("metrics.box.map: ", metrics.box.map)
print("metrics.box.map50: ", metrics.box.map50)
print("metrics.box.map75: ", metrics.box.map75)
