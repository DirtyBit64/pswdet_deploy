# pswdet_deploy
道路抛洒物检测 ———— 基于仓库[YOLOv8](https://github.com/ultralytics/ultralytics)进行订制化开发，针对抛洒物检测误报过多等问题，嵌入分割过滤模块PIDNet(用于排除道路区域外的误报)和分类过滤模块ResNet(用于排除道路内的误报)。

## Install
```bash
pip install -r requirements.txt
```

## :label: TODO 
- [x] 添加分割过滤逻辑
- [ ] 添加分类过滤逻辑

## Extended arguments
- ```seg_filter``` : 是否启用分割过滤模块，若开启需将道路分割模型权重如pidnet.onnx置于'ultralytics/models/yolo/detect/roadseg/*.onnx'路径下
- ```save_if_box_father``` : 仅当```save=false```时，开启该参数有效，实现只保存检测出目标的结果集

## Contributing
PRs accepted.

## License
**nuist** © **ailab-cv**
