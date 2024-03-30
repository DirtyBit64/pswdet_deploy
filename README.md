# pswdet_deploy
道路抛洒物检测 ———— 基于仓库[YOLOv8](https://github.com/ultralytics/ultralytics)进行订制化开发，针对抛洒物检测误报过多等问题，加入分割过滤模块PIDNet(用于排除道路区域外的误报)和分类过滤模块ResNet(用于排除道路内的误报)。

## Install
```bash
pip install -r requirements.txt
```

## :label: TODO 
- [x] 添加分割过滤逻辑
- [ ] 添加分类过滤逻辑

## Contributing
PRs accepted.

## License
**Nuist** © **ailab-cv**
