#include <fstream>

#include <nlohmann/json.hpp>

#include "ncnn_yolo.h"


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <task> <model_path> <image_path>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string task = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];

    NCNNRunner ncnn_runner(model_path, task);
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Could not read image: " << image_path << std::endl;
        return EXIT_FAILURE;
    }
    auto results = ncnn_runner.infer(img);
    std::cout << "[INFO] Detection results count: " << results.size() << std::endl;

    // get results
    nlohmann::json json;
	std::ifstream jfile("config.json");
	jfile >> json;
    std::vector<std::string> class_names = json[task]["class_names"].get<std::vector<std::string>>();
    cv::Point cls_initpos(10, 20); // 分类文本位置
    for(const auto& res:results){
        std::string label = class_names[res.class_id] + ": " + cv::format("%.2f", res.score);
        std::cout << "[RESULT] " << label << std::endl;
        if (task == "cls") {
            cv::putText(img, label, cls_initpos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(img, label, cls_initpos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            cls_initpos = cv::Point(cls_initpos.x, cls_initpos.y + 20);
        }
        else if (task == "det"){
            cv::rectangle(img, cv::Point(res.boxes[0][0], res.boxes[0][1]), cv::Point(res.boxes[1][0], res.boxes[1][1]), cv::Scalar(0, 255, 0), 2);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        else if (task == "obb"){
            std::vector<cv::Point> points;
            for (const auto& box : res.boxes) {
                points.emplace_back(cv::Point(box[0], box[1]));
            }
            const std::vector<std::vector<cv::Point>> pts{points};
            cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), 2);
        }
        else if (task == "pose") {
            cv::rectangle(img, cv::Point(res.boxes[0][0], res.boxes[0][1]), cv::Point(res.boxes[1][0], res.boxes[1][1]), cv::Scalar(0, 255, 0), 2);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            std::vector<std::vector<int>> skeleton = json[task]["skeleton"].get<std::vector<std::vector<int>>>();
            for (const auto& kp : res.keypoints) {
                if (kp.pt.pt.x >= 0 && kp.pt.pt.y >= 0) {
                    cv::circle(img, cv::Point(static_cast<int>(kp.pt.pt.x), static_cast<int>(kp.pt.pt.y)), 3, cv::Scalar(0, 0, 255), -1);
                }
            }
            for (const auto& sk : skeleton) {
                const auto& pt1 = res.keypoints[sk[0]].pt.pt;
                const auto& pt2 = res.keypoints[sk[1]].pt.pt;
                if (pt1.x >= 0 && pt1.y >= 0 && pt2.x >= 0 && pt2.y >= 0) {
                    cv::line(img, cv::Point(static_cast<int>(pt1.x), static_cast<int>(pt1.y)),
                             cv::Point(static_cast<int>(pt2.x), static_cast<int>(pt2.y)), cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        else if (task == "seg") {
            cv::rectangle(img, cv::Point(res.boxes[0][0], res.boxes[0][1]), cv::Point(res.boxes[1][0], res.boxes[1][1]), cv::Scalar(0, 255, 0), 2);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(img, label, cv::Point(res.boxes[0][0], res.boxes[0][1]-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);            
            std::vector<cv::Point> single_contour;
            for (const auto& point_vec : res.contour) {
                if(point_vec.size() >= 2) {
                     single_contour.emplace_back(cv::Point(point_vec[0], point_vec[1]));
                }
            }
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(single_contour);
            cv::drawContours(img, contours, -1, cv::Scalar(255, 0, 0), 2);
        }
    }
    // cv::imwrite("result.jpg", img);
    return EXIT_SUCCESS;
}