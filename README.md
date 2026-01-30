# ncnn runner
## convert model
`ultralytics == 8.4.7` 内置了 `ncnn` 导出功能，可以直接运行下面命令下载并转换模型：
```bash
pip install ultralytics ncnn
python ./script/download_model.py
```

## python
```bash
# YOLOv8
python ./python/yolo_runner.py cls models/yolov8n-cls_ncnn_model images/bus.jpg
python ./python/yolo_runner.py det models/yolov8n_ncnn_model images/bus.jpg
python ./python/yolo_runner.py obb models/yolov8n-obb_ncnn_model images/P0015.jpg
python ./python/yolo_runner.py seg models/yolov8n-seg_ncnn_model images/bus.jpg
python ./python/yolo_runner.py pose models/yolov8n-pose_ncnn_model images/zidane.jpg

# YOLO11
python ./python/yolo_runner.py cls models/yolo11n-cls_ncnn_model images/bus.jpg
python ./python/yolo_runner.py det models/yolo11n_ncnn_model images/bus.jpg
python ./python/yolo_runner.py obb models/yolo11n-obb_ncnn_model images/P0015.jpg
python ./python/yolo_runner.py seg models/yolo11n-seg_ncnn_model images/bus.jpg
python ./python/yolo_runner.py pose models/yolo11n-pose_ncnn_model images/zidane.jpg

# YOLO12
python ./python/yolo_runner.py det models/yolo12n_ncnn_model images/bus.jpg
```