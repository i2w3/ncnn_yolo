#ifndef NCNN_YOLO_H
#define NCNN_YOLO_H

#include <string>
#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>


struct Keypoint {
    cv::KeyPoint pt;
    float score;
};


struct YOLOInput {
    cv::Mat image;
    ncnn::Mat blob;
    int top;
    int bottom;
    int left;
    int right;
    float scale;

    YOLOInput(const cv::Mat& _image, ncnn::Mat& _blob, int _top, int _bottom, int _left, int _right, float _scale)
        : image(_image), blob(_blob), top(_top), bottom(_bottom), left(_left), right(_right), scale(_scale) {}
};


struct YOLOOutput {
    int class_id;
    std::vector<std::vector<int>> boxes; // [[x1, y1], [x2, y2], ...]
    float score;
    std::vector<std::vector<int>> contour;
    std::vector<Keypoint> keypoints;

    YOLOOutput() = default; // 保留默认无参构造函数
    YOLOOutput(const int& _class_id, const std::vector<std::vector<int>>& _boxes, const float& _score,
               const std::vector<std::vector<int>>& _contour, const std::vector<Keypoint>& _keypoints)
        : class_id(_class_id), boxes(_boxes), score(_score), contour(_contour), keypoints(_keypoints) {}
};


class NCNNRunner {
public:
    NCNNRunner(const std::string& _model_path, const std::string& _task);
    ~NCNNRunner();

    std::vector<YOLOOutput> infer(const cv::Mat& img);

    const std::string task;
private:
    ncnn::Net net;
    const float conf_threshold = 0.25;
    const float iou_threshold  = 0.45;
    const float mask_threshold = 0.5;
    const float kps_threshold  = 0.5;
    const int kps = 17; // 硬编码关键点数量
    const int max_det = 100;
    int size;

    YOLOInput preprocess(const cv::Mat& img);
    std::vector<YOLOOutput> postprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::vector<YOLOOutput> clspostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::vector<YOLOOutput> detpostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::vector<YOLOOutput> obbpostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::vector<YOLOOutput> posepostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::vector<YOLOOutput> segpostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input);
    std::unordered_map<std::string, ncnn::Mat> ncnn_infer(const ncnn::Mat& in_blob);
};


inline std::string get_mat_shape(const ncnn::Mat& m) { // inline 避免多重定义
    if (m.dims == 1) {
        return "[w=" + std::to_string(m.w) + "]";
    } else if (m.dims == 2) {
        return "[w=" + std::to_string(m.w) + ", h=" + std::to_string(m.h) + "]";
    } else if (m.dims == 3) {
        return "[w=" + std::to_string(m.w) + ", h=" + std::to_string(m.h) + ", c=" + std::to_string(m.c) + "]";
    } else if (m.dims == 4) {
        return "[w=" + std::to_string(m.w) + ", h=" + std::to_string(m.h) + ", d=" + std::to_string(m.d) + ", c=" + std::to_string(m.c) + "]";
    } else {
        return "Unknown shape dimension";
    }
}

#endif // NCNN_YOLO_H