#include "ncnn_yolo.h"

std::vector<YOLOOutput> NCNNRunner::segpostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input){
    std::vector<YOLOOutput> detections;

    ncnn::Mat ncnn_output = outputs.at("out0");
    ncnn::Mat ncnn_protos = outputs.at("out1");
    cv::Mat cv_output = cv::Mat((int)ncnn_output.h, (int) ncnn_output.w, CV_32F, (float *)ncnn_output.data);

    const int proto_h = ncnn_protos.h;
    const int proto_w = ncnn_protos.w;
    const int proto_area = proto_h * proto_w;
    cv::Mat cv_protos = cv::Mat(32, proto_area, CV_32F, (float *)ncnn_protos.data); // [proto_h * proto_w, 32]

    // Transpose if the output is [Features x Anchors] (84 x 8400) -> [Anchors x Features] (8400 x 84)
    if (cv_output.cols > cv_output.rows) {
        cv::transpose(cv_output, cv_output);
    }

    const int num_classes = cv_output.cols - 4 - 32;

    // 预分配 vector 以避免频繁的内存重新分配
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    boxes.reserve(cv_output.rows);
    confidences.reserve(cv_output.rows);
    class_ids.reserve(cv_output.rows);
    cv::Mat masks_coeffs; // 保存过滤后的 mask coefficients

    // 处理每个检测框
    for (int i = 0; i < cv_output.rows; i++) {
        const float *row_ptr = cv_output.row(i).ptr<float>();
        const float *bboxes_ptr = row_ptr;
        const float *classes_ptr = row_ptr + 4;
        const float *max_s_ptr = std::max_element(classes_ptr, classes_ptr + num_classes);
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

            // 保存 mask conefficients
            cv::Mat mask_coeff = cv::Mat(1, 32, CV_32F);
            // row_ptr + 4 + num_classes cannot be used if data is not continuous or just copied directly as pointer might be risky.
            // Copy data safely.
            const float* mask_ptr = row_ptr + 4 + num_classes;
            for (int k = 0; k < 32; k++) {
                mask_coeff.at<float>(0, k) = mask_ptr[k];
            }
            masks_coeffs.push_back(mask_coeff); 
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
    detections.reserve(indices.size()); // 预分配最终结果vector
    cv::Mat filtered_masks_coeffs;      // 保存NMS后对应的mask系数
    for (int idx : indices) {
        YOLOOutput det;
        det.boxes = {{boxes[idx].x                   , boxes[idx].y}, 
                     {boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height}};
        det.score = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
        
        filtered_masks_coeffs.push_back(masks_coeffs.row(idx));
    }

    // 计算每个检测框的分割掩码
    cv::Mat matmulRes = (filtered_masks_coeffs * cv_protos).t();
    cv::Mat maskMat = matmulRes.reshape(static_cast<int>(indices.size()), {proto_w, proto_h}); // [num_boxes, proto_h, proto_w]
    std::vector<cv::Mat> maskChannels;
    maskChannels.reserve(indices.size());
    cv::split(maskMat, maskChannels);

    for (size_t i = 0; i < detections.size(); i++) {
        // 1. 获取原图坐标系下的 Box (同时修改 detections 里的 box)
        // 注意：Input Size 下的 box 仍然保留在 boxes[indices[i]] 中
        cv::Rect input_box = boxes[indices[i]];
        
        // 调整边界框以补偿前处理中的缩放和填充
        for (auto& box : detections[i].boxes) {
            box[0] = static_cast<int>((box[0] - yolo_input.left) / yolo_input.scale);
            box[1] = static_cast<int>((box[1] - yolo_input.top) / yolo_input.scale);
        }
        // 简单的边界截断
        int img_h = yolo_input.image.rows;
        int img_w = yolo_input.image.cols;
        auto& box_pt1 = detections[i].boxes[0];
        auto& box_pt2 = detections[i].boxes[1];
        box_pt1[0] = std::max(0, std::min(box_pt1[0], img_w));
        box_pt1[1] = std::max(0, std::min(box_pt1[1], img_h));
        box_pt2[0] = std::max(0, std::min(box_pt2[0], img_w));
        box_pt2[1] = std::max(0, std::min(box_pt2[1], img_h));

        int box_w = box_pt2[0] - box_pt1[0];
        int box_h = box_pt2[1] - box_pt1[1];
        if (box_w <= 0 || box_h <= 0) continue;

        // mask 处理
        cv::Mat dest;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest); // [proto_h, proto_w]

        // Resize mask to input size (e.g. 640x640)
        cv::Mat mask;
        cv::resize(dest, mask, cv::Size(this->size, this->size));

        // Crop mask by input box
        // 边界保护
        int bx = std::max(0, input_box.x);
        int by = std::max(0, input_box.y);
        int bw = std::min(input_box.width, this->size - bx);
        int bh = std::min(input_box.height, this->size - by);
        
        if (bw <= 0 || bh <= 0) continue;
        
        cv::Mat mask_crop = mask(cv::Rect(bx, by, bw, bh));

        // Resize crop to original box size
        cv::resize(mask_crop, mask_crop, cv::Size(box_w, box_h), 0, 0, cv::INTER_NEAREST);

        // Threshold
        cv::Mat bin_mask;
        cv::threshold(mask_crop, bin_mask, 0.5, 255, cv::THRESH_BINARY);
        bin_mask.convertTo(bin_mask, CV_8UC1);

        // Find Contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Select largest contour and offset it
        if (!contours.empty()) {
            // Find largest contour
            size_t max_idx = 0;
            double max_area = 0;
            for(size_t k=0; k<contours.size(); ++k){
                double area = cv::contourArea(contours[k]);
                if(area > max_area){
                    max_area = area;
                    max_idx = k;
                }
            }

            detections[i].contour.reserve(contours[max_idx].size());
            for (const auto& pt : contours[max_idx]) {
                // Offset by original box top-left
                detections[i].contour.push_back({pt.x + box_pt1[0], pt.y + box_pt1[1]});
            }
        } else {
            detections[i].contour = {};
        }
    }
    return detections;
}
