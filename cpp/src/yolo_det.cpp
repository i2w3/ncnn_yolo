#include "ncnn_yolo.h"

std::vector<YOLOOutput> NCNNRunner::detpostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input){
    std::vector<YOLOOutput> detections;

    ncnn::Mat ncnn_output = outputs.at("out0");
    cv::Mat cv_output = cv::Mat((int)ncnn_output.h, (int) ncnn_output.w, CV_32F, (float *)ncnn_output.data);
    
    // Transpose if the output is [Features x Anchors] (84 x 8400) -> [Anchors x Features] (8400 x 84)
    if (cv_output.cols > cv_output.rows) {
        cv::transpose(cv_output, cv_output);
    }

    // 预分配 vector 以避免频繁的内存重新分配
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(cv_output.rows);
    confidences.reserve(cv_output.rows);
    class_ids.reserve(cv_output.rows);

    // 处理每个检测框
    for (int i = 0; i < cv_output.rows; i++) {
        const float *row_ptr = cv_output.row(i).ptr<float>();
        const float *bboxes_ptr = row_ptr;
        const float *classes_ptr = row_ptr + 4;
        const float *max_s_ptr = std::max_element(classes_ptr, classes_ptr + cv_output.cols - 4);
        float score = *max_s_ptr;

        if (score > this->conf_threshold) {
            float xc = *bboxes_ptr++;
            float yc = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x = xc - w / 2.0f;
            float y = yc - h / 2.0f;
            
            boxes.emplace_back(cv::Rect2f(x, y, w, h));
            confidences.emplace_back(score);
            class_ids.emplace_back(max_s_ptr - classes_ptr);
        }
    }
    
    // 应用非极大值抑制(NMS)消除冗余的重叠框
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.0f, this->iou_threshold, indices, 1.f, this->max_det);
    }
    else {
        return detections; // 没有检测到目标，直接返回空结果
    }

    // 将过滤后的索引转换为最终的YOLODetection对象
    detections.reserve(indices.size());  // 预分配最终结果vector
    for (int idx : indices) {
        YOLOOutput det;
        det.boxes = {{boxes[idx].x                   , boxes[idx].y}, 
                     {boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height}};
        det.score = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    for (auto& det : detections) {
        // 调整边界框以补偿前处理中的缩放和填充
        for (auto& box : det.boxes) {
            box[0] = static_cast<int>((box[0] - yolo_input.left) / yolo_input.scale);
            box[1] = static_cast<int>((box[1] - yolo_input.top) / yolo_input.scale);
        }
    }
    return detections;
}
