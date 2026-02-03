#include "ncnn_yolo.h"

std::vector<YOLOOutput> NCNNRunner::clspostprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input){
    auto _ = yolo_input; // 未使用参数，避免编译警告
    const int top_k = 5;

    std::vector<YOLOOutput> detections;
    detections.resize(top_k);

    ncnn::Mat ncnn_output = outputs.at("out0");
    const int size = ncnn_output.w;

    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(ncnn_output[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end(),
                      std::greater<std::pair<float, int> >());

    for (int i = 0; i < top_k; i++) {
        detections[i].class_id = vec[i].second;
        detections[i].score = vec[i].first;
    }
    return detections;

}