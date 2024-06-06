from ultralytics import YOLO

# model = YOLO('weights/once2.pt')
# # model.export(format='engine', imgsz=(1280, 1280), device=0)
# model.export(format='onnx', imgsz=(736, 1280), half=False, device=0)


import torch.nn

model = torch.load(r'D:\CV\PJclassfy\weights\17_res.pth', map_location='cuda')
model.to('cuda')
model.eval()

input_names = ['input']
output_names = ['output']


x = torch.randn(1, 3, 64, 64, requires_grad=True).to('cuda')

torch.onnx.export(model, x, 'weights/test1.onnx', input_names=input_names, output_names=output_names, verbose='True')
