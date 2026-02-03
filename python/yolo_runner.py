import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Union

import cv2
import numpy as np
import ncnn


def parse_args():
    parser = argparse.ArgumentParser(description="NCNN RUNNER")
    parser.add_argument("task", type=str, help="YOLO 任务类型：分类(cls)、检测(det)、定向检测(obb)、分割(seg)、姿态估计(pose)")
    parser.add_argument("model_path", type=str, help="YOLO 模型文件路径")
    parser.add_argument("image_path", type=str, help="输入图像文件路径")
    return parser.parse_args()


@dataclass
class YOLOInput:
    image: np.ndarray
    blob: Union[np.ndarray, ncnn.Mat]
    top: int
    bottom: int
    left: int
    right: int
    scale: float


@dataclass
class Keypoint:
    pt: cv2.KeyPoint
    score: float


@dataclass
class YOLOOutput:
    class_id: int
    box: List[List[int]] # [[x1. y1], [x2, y2], ...]
    score: float
    contour: List[np.ndarray] = field(default_factory=list)
    keypoints: List[Keypoint] = field(default_factory=list)


class NCNNRunner:
    def __init__(self, model_path: Union[str, Path], task: str):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path
        self.task = task
        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.opt.use_vulkan_compute = False # 根据需要启用 Vulkan 加速，否则使用 CPU
        r1 = self.net.load_param(str(self.model_path / "model.ncnn.param"))
        r2 = self.net.load_model(str(self.model_path / "model.ncnn.bin"))

        if r1 == 0 and r2 == 0:
            print("[INFO] NCNN model loaded successfully.")
        else:
            raise RuntimeError(f"Failed to load NCNN model: param_ret={r1}, model_ret={r2}")

        if self.task in ["det", "seg", "pose"]:
            self.size = 640
        elif self.task == "cls":
            self.size = 224
        elif self.task == "obb":
            self.size = 1024
        else:
            raise ValueError(f"Unsupported task type: {self.task}")
        
        self.conf_threshold = 0.25
        self.iou_threshold  = 0.45
        self.mask_threshold = 0.5
        self.kps_threshold  = 0.5
        self.max_det = 300

        # 仅硬编码 pose 关键点数量, num_classes 和 mask_protos_size 在推理时动态获取
        self.kps = 17

        # Warm up 并获取输出数量
        print("[INFO] Warming up the NCNN model...")
        dummy_input = np.zeros((3, self.size, self.size), dtype=np.float32) # 注意 model.export 使用 half=True 时，dtype=np.float16

        outs = []
        print("[INFO] NCNN model outputs: ")
        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(dummy_input))
            for i in self.net.output_names():
                ret, out = ex.extract(i)
                if ret == 0:
                    np_out = np.array(out)
                    print(f"\t{i}: {np_out.shape}")
                    outs.append(np.array(out))
        self.out_len = len(outs)
        print("[INFO] NCNN model output shapes: " + ", ".join([str(o.shape) for o in outs]))
        print("[INFO] Model warm up done.")
    
    def preProcess(self, image_data:np.ndarray) -> np.ndarray:
        # LetterBox mode
        img_h, img_w = image_data.shape[:2]
        scale = self.size / max(img_h, img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        
        pad_w = (self.size - new_w) / 2.
        pad_h = (self.size - new_h) / 2.

        top = int(pad_h - 0.1)
        bottom = int(pad_h + 0.1)
        left = int(pad_w - 0.1)
        right = int(pad_w + 0.1)

        # 1. ncnn mat
        ncnn_img = ncnn.Mat.from_pixels_resize(image_data, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, new_w, new_h)
        blob_img = ncnn.copy_make_border(ncnn_img, top, bottom, left, right, ncnn.BorderType.BORDER_CONSTANT, 114.0)
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        blob_img.substract_mean_normalize(mean_vals, norm_vals)

        # 2. numpy array
        # resized_img = cv2.resize(image_data, (new_w, new_h))
        # pad_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]) # 灰色填充
        # blob_img = cv2.dnn.blobFromImage(pad_img, 1/255.0, (self.size, self.size), swapRB=True, crop=False, ddepth=cv2.CV_32F)

        return YOLOInput(image=image_data, blob=blob_img, top=top, bottom=bottom, left=left, right=right, scale=scale)
    
    def ncnn_infer(self, input_blob:Union[np.ndarray, ncnn.Mat]) -> List[np.ndarray]:
        outs = []
        with self.net.create_extractor() as ex:
            if type(input_blob) is np.ndarray:
                input_mat = ncnn.Mat(input_blob[0]) # (3, h, w)
            elif type(input_blob) is ncnn.Mat:
                input_mat = ncnn.Mat(input_blob)
            else:
                raise TypeError(f"Unsupported input_blob type: {type(input_blob)}")
            ex.input("in0", input_mat)
            
            for i in range(self.out_len):
                ret, out = ex.extract(f"out{i}")
                outs.append(np.array(out)) # 注意这里不会再检查 ret 值
        return outs
    
    def infer(self, image_data:np.ndarray):
        start_time = time.time()
        input_blob = self.preProcess(image_data)
        model_outs = self.ncnn_infer(input_blob.blob)
        results = self.postProcess(model_outs, input_blob)
        end_time = time.time()
        print(f"[INFO] Inference time: {end_time - start_time:.3f} seconds")
        return results

    def postProcess(self, *args, **kwargs) -> List[YOLOOutput]:
        if self.task == "cls":
            return self.clsProcess(*args, **kwargs)
        elif self.task == "det":
            return self.detProcess(*args, **kwargs)
        elif self.task == "obb":
            return self.obbProcess(*args, **kwargs)
        elif self.task == "seg":
            return self.segProcess(*args, **kwargs)
        elif self.task == "pose":
            return self.poseProcess(*args, **kwargs)
        # 模型加载的时候已经做了 self.task 检查，后续都不会有 else 分支
    
    def clsProcess(self, model_outs:List[np.ndarray], yolo_input:YOLOInput) -> List[YOLOOutput]:
        # softmax + top 5
        scores = model_outs[0]
        ## print(scores.shape) # [1000]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        top5_ids = probs.argsort()[-5:][::-1]
        results = []
        for class_id in top5_ids:
            results.append(YOLOOutput(class_id=int(class_id), box=[], score=float(probs[class_id])))
        return results
    
    def detProcess(self, model_outs:List[np.ndarray], yolo_input:YOLOInput) -> List[YOLOOutput]:
        predictions = model_outs[0] # shape: (84, 8400)
        predictions = predictions.transpose(1, 0) # shape: (8400, 84)
        class_ids = []
        boxes = []
        confidences = []
        
        for pred in predictions:
            score = pred[4:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > self.conf_threshold:
                # 最好先做个置信度过滤，减少 NMS 计算量
                cx, cy, w, h = pred[0:4]
                x = int(cx - w / 2.0)
                y = int(cy - h / 2.0)
                boxes.append([x, y, int(w), int(h)])
                class_ids.append(class_id)
                confidences.append(float(confidence))
        # 注意不是按类别 nms，而是所有类别一起做 nms
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        results = []
        img_h, img_w = yolo_input.image.shape[:2]
        for i in indices[:self.max_det]:
            box = boxes[i] # [x, y, w, h] 在 640x640 画布上的坐标
            score = confidences[i]
            class_id = class_ids[i]
            
            # 移除 padding (top, left) + 缩放回原图 (scale)
            x1 = (box[0] - yolo_input.left) / yolo_input.scale
            y1 = (box[1] - yolo_input.top) / yolo_input.scale
            x2 = (box[0] + box[2] - yolo_input.left) / yolo_input.scale
            y2 = (box[1] + box[3] - yolo_input.top) / yolo_input.scale

            # 边界截断
            x1 =  max(0, min(x1, img_w))
            y1 =  max(0, min(y1, img_h))
            x2 =  max(0, min(x2, img_w))
            y2 =  max(0, min(y2, img_h))

            results.append(YOLOOutput(box=[[int(x1), int(y1)], [int(x2), int(y2)]], score=score, class_id=class_id))
        return results
    
    def obbProcess(self, model_outs:List[np.ndarray], yolo_input:YOLOInput) -> List[YOLOOutput]:
        predictions = model_outs[0] # shape: (84, 8400)
        predictions = predictions.transpose(1, 0) # shape: (8400, 84)
        class_ids = []
        boxes = []
        confidences = []

        for pred in predictions:
            score = pred[4:-1] # obb 最后一个是角度
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > self.conf_threshold:
                # 最好先做个置信度过滤，减少 NMS 计算量
                cx, cy, w, h = pred[0:4]
                rad = pred[-1]
                angle = rad * 180.0 / np.pi  # 转为角度制
                boxes.append([[cx, cy], [int(w), int(h)], angle])
                class_ids.append(class_id)
                confidences.append(float(confidence))
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_threshold, self.iou_threshold)
        
        results = []
        img_h, img_w = yolo_input.image.shape[:2]
        for i in indices[:self.max_det]:
            box = cv2.boxPoints(boxes[i]).tolist() # 四个点坐标列表
            score = confidences[i]
            class_id = class_ids[i]
            
            scale_box = []
            for point in box:
                # 移除 padding (top, left) + 缩放回原图 (scale)
                x1 = (point[0] - yolo_input.left) / yolo_input.scale
                y1 = (point[1] - yolo_input.top) / yolo_input.scale

                # 边界截断
                x1 =  max(0, min(x1, img_w))
                y1 =  max(0, min(y1, img_h))
                scale_box.append([int(x1), int(y1)])

            results.append(YOLOOutput(box=scale_box, score=score, class_id=class_id))
        return results
    
    def segProcess(self, model_outs:List[np.ndarray], yolo_input:YOLOInput) -> List[YOLOOutput]:
        predictions = model_outs[0] # shape: (116, 8400)
        predictions = predictions.transpose(1, 0) # shape: (8400, 116)
        protos = model_outs[1]     # shape: (32, 160, 160)
        proto_h, proto_w = protos.shape[1:] # 160, 160
        class_ids = []
        boxes = []
        confidences = []
        mask_coffs = []

        for pred in predictions:
            score = pred[4:-32]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > self.conf_threshold:
                # 最好先做个置信度过滤，减少 NMS 计算量
                cx, cy, w, h = pred[0:4]
                x = int(cx - w / 2.0)
                y = int(cy - h / 2.0)
                boxes.append([x, y, int(w), int(h)])
                class_ids.append(class_id)
                confidences.append(float(confidence))
                mask_coffs.append(pred[-32:])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        results = []
        img_h, img_w = yolo_input.image.shape[:2]
        for i in indices[:self.max_det]:
            box = boxes[i] # [x, y, w, h] 在 640x640 画布上的坐标
            score = confidences[i]
            class_id = class_ids[i]
            
            # 移除 padding (top, left) + 缩放回原图 (scale)
            x1 = (box[0] - yolo_input.left) / yolo_input.scale
            y1 = (box[1] - yolo_input.top) / yolo_input.scale
            x2 = (box[0] + box[2] - yolo_input.left) / yolo_input.scale
            y2 = (box[1] + box[3] - yolo_input.top) / yolo_input.scale

            # 边界截断
            x1 =  max(0, min(x1, img_w))
            y1 =  max(0, min(y1, img_h))
            x2 =  max(0, min(x2, img_w))
            y2 =  max(0, min(y2, img_h))

            # proto 是全图的，需要裁剪并缩放到 bbox 大小
            box_w = int(x2 - x1)
            box_h = int(y2 - y1)
            if box_w <= 0 or box_h <= 0:
                continue

            # 生成掩码
            mask_coeff = np.array(mask_coffs[i])
            mask = np.dot(protos.reshape(32, -1).T, mask_coeff).reshape(proto_h, proto_w)
            mask = 1 / (1 + np.exp(-mask))  # sigmoid

            mask = cv2.resize(mask, (self.size, self.size))
            bx, by, bw, bh = box[0], box[1], box[2], box[3]
            # 简单边界保护
            bx = max(0, bx)
            by = max(0, by)
            bw = min(bw, self.size - bx)
            bh = min(bh, self.size - by)
            
            mask_crop = mask[by:by+bh, bx:bx+bw]
            mask_crop = cv2.resize(mask_crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
            mask_crop = (mask_crop > self.mask_threshold).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 合并所有轮廓点并平移到原图坐标
            final_contours = []
            for cnt in contours:
                # cnt 是 (N, 1, 2) 的数组
                cnt = cnt.astype(np.int32)
                cnt += np.array([int(x1), int(y1)], dtype=np.int32)
                final_contours.append(cnt)
            largest_contour = max(final_contours, key=cv2.contourArea) # 选择最大轮廓的 cnt
            results.append(YOLOOutput(box=[[int(x1), int(y1)], [int(x2), int(y2)]], score=score, class_id=class_id, contour=largest_contour))
        return results
    
    def poseProcess(self, model_outs:List[np.ndarray], yolo_input:YOLOInput) -> List[YOLOOutput]:
        predictions = model_outs[0] # shape: (56, 8400)
        predictions = predictions.transpose(1, 0) # shape: (8400, 56)
        class_ids = []
        boxes = []
        confidences = []
        all_kps = []

        num_classes = predictions.shape[1] - 4 - (3 * self.kps)
        
        for pred in predictions:
            score = pred[4:4+num_classes]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > self.conf_threshold:
                # 最好先做个置信度过滤，减少 NMS 计算量
                cx, cy, w, h = pred[0:4]
                x = int(cx - w / 2.0)
                y = int(cy - h / 2.0)
                boxes.append([x, y, int(w), int(h)])
                class_ids.append(class_id)
                confidences.append(float(confidence))
                kps = []
                for i in range(self.kps):
                    kps.append(Keypoint(pt=cv2.KeyPoint(x=pred[4+num_classes + i*3], 
                                                        y=pred[4+num_classes + i*3 +1], 
                                                        size=1),          
                                        score=pred[4+num_classes + i*3 +2]))
                all_kps.append(kps)
        # 注意不是按类别 nms，而是所有类别一起做 nms
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        results = []
        img_h, img_w = yolo_input.image.shape[:2]
        for i in indices[:self.max_det]:
            box = boxes[i] # [x, y, w, h] 在 640x640 画布上的坐标
            score = confidences[i]
            class_id = class_ids[i]
            kps:List[Keypoint] = all_kps[i]
            
            # 移除 padding (top, left) + 缩放回原图 (scale)
            x1 = (box[0] - yolo_input.left) / yolo_input.scale
            y1 = (box[1] - yolo_input.top) / yolo_input.scale
            x2 = (box[0] + box[2] - yolo_input.left) / yolo_input.scale
            y2 = (box[1] + box[3] - yolo_input.top) / yolo_input.scale

            # 边界截断
            x1 =  max(0, min(x1, img_w))
            y1 =  max(0, min(y1, img_h))
            x2 =  max(0, min(x2, img_w))
            y2 =  max(0, min(y2, img_h))

            scale_kps = []
            for kp in kps:
                if kp.score < self.kps_threshold:
                    kp.pt.pt = (-1, -1) # 无效点, 不过后面画图还是用 score 判断
                else:
                    kp.pt.pt = ((kp.pt.pt[0] - yolo_input.left) / yolo_input.scale, (kp.pt.pt[1] - yolo_input.top) / yolo_input.scale)
                scale_kps.append(kp)

            results.append(YOLOOutput(box=[[int(x1), int(y1)], [int(x2), int(y2)]], score=score, class_id=class_id, keypoints=scale_kps))
        return results

if __name__ == "__main__":
    args = parse_args()
    model = NCNNRunner(args.model_path, args.task)
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    results = model.infer(image)

    # get results
    import json
    with open("config.json", "r") as f:
        config = json.load(f)
    class_names = config[args.task]["class_names"]
    cls_initpos = (10, 20) # 分类文本位置
    for res in results:
        label = f"{class_names[res.class_id]}: {res.score:.2f}"
        print(f"[RESULT] {label}")
        if args.task == "cls":
            cv2.putText(image, label, cls_initpos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(image, label, cls_initpos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cls_initpos = (cls_initpos[0], cls_initpos[1] + 20)
        elif args.task == "det":
            box = res.box
            cv2.rectangle(image, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 2)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif args.task == "obb":
            box = res.box
            cv2.polylines(image, [np.array(box, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        elif args.task == "pose":
            box = res.box
            cv2.rectangle(image, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 2)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            skeleton = config[args.task]["skeleton"]
            for kp in res.keypoints:
                if kp.score >= model.kps_threshold:
                    cv2.circle(image, (int(kp.pt.pt[0]), int(kp.pt.pt[1])), 3, (0, 0, 255), -1)
            for sk in skeleton:
                pt1 = res.keypoints[sk[0]].pt.pt
                pt2 = res.keypoints[sk[1]].pt.pt
                if res.keypoints[sk[0]].score >= model.kps_threshold and res.keypoints[sk[1]].score >= model.kps_threshold:
                    cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
        elif args.task == "seg":
            box = res.box
            cv2.rectangle(image, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 2)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(image, label, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.drawContours(image, [res.contour], -1, (255, 0, 0), 2)
    cv2.imwrite(f"./assets/ncnn_{Path(args.model_path).stem.split('_')[0]}.jpg", image)