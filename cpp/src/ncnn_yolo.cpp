#include "ncnn_yolo.h"

NCNNRunner::NCNNRunner(const std::string& _model_path, const std::string& _task)
    : task(_task) {
    // Constructor implementation
    std::filesystem::path model_path(_model_path);
    this->net.opt.num_threads = 4;
    this->net.opt.use_vulkan_compute = false;
    int r1 = this->net.load_param((model_path / "model.ncnn.param").string().c_str());
    int r2 = this->net.load_model((model_path / "model.ncnn.bin").string().c_str());
    if(r1 == 0 && r2 == 0) {
        std::cout << "[INFO] NCNN model loaded successfully." << std::endl;
    } else {
        throw std::runtime_error("[ERROR] Failed to load NCNN model files.");
    }
    if (_task == "cls"){
        this->size = 224;
    } else if (_task == "det" || _task == "seg" || _task == "pose") {
        this->size = 640;
    } else if (_task == "obb") {
        this->size = 1024;
    } else {
        throw std::invalid_argument("[ERROR] Unsupported task type: " + _task);
    }
    // warm up
    std::cout << "[INFO] NCNN model outputs: " << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat dummy_input(this->size, this->size, CV_8UC3, cv::Scalar(0, 0, 0));
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dummy_input.data, ncnn::Mat::PIXEL_BGR, dummy_input.cols, dummy_input.rows, this->size, this->size);
    ncnn::Extractor ex = this->net.create_extractor();
    ex.input("in0", in);
    for(const auto& output_name : this->net.output_names()) {
        ncnn::Mat out;
        ex.extract(output_name, out);
        std::cout << " - " << output_name << ": " << get_mat_shape(out) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> warmup_time = end - start;
    std::cout << "[INFO] Warmup time: " << warmup_time.count() << " ms" << std::endl;
}

NCNNRunner::~NCNNRunner() {
    // Destructor implementation
    this->net.clear();
}

YOLOInput NCNNRunner::preprocess(const cv::Mat& img) {
    int img_w = img.cols;
    int img_h = img.rows;
    float scale = this->size / (float)std::max(img_w, img_h);
    int new_w = (int)(img_w * scale);
    int new_h = (int)(img_h * scale);

    float pad_w = (this->size - new_w) / 2.0f;
    float pad_h = (this->size - new_h) / 2.0f;
    
    int top     = (int)std::round(pad_h - 0.1f);
    int bottom  = (int)std::round(pad_h + 0.1f);
    int left    = (int)std::round(pad_w - 0.1f);
    int right   = (int)std::round(pad_w + 0.1f);

    // 1. ncnn mat
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PixelType::PIXEL_BGR2RGB, img_w, img_h, new_w, new_h);
    ncnn::Mat blob_img;
    ncnn::copy_make_border(ncnn_img, blob_img, top, bottom, left, right, ncnn::BorderType::BORDER_CONSTANT, 114.f);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    blob_img.substract_mean_normalize(0, norm_vals);

    // 2. cv mat
    // cv::Mat resized_img;
    // cv::resize(img, resized_img, cv::Size(new_w, new_h));

    return YOLOInput(img, blob_img, top, bottom, left, right, scale);
}

std::vector<YOLOOutput> NCNNRunner::infer(const cv::Mat& img){
    auto start = std::chrono::high_resolution_clock::now();
    auto yolo_input = preprocess(img);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> process_time = end - start;
    std::cout << "[INFO] PreProcess time: " << process_time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto outputs = ncnn_infer(yolo_input.blob);
    end = std::chrono::high_resolution_clock::now();
    process_time = end - start;
    std::cout << "[INFO] Infer time: " << process_time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto results = postprocess(outputs, yolo_input);
    end = std::chrono::high_resolution_clock::now();
    process_time = end - start;
    std::cout << "[INFO] PostProcess time: " << process_time.count() << " ms" << std::endl;

    return results;
}

std::unordered_map<std::string, ncnn::Mat> NCNNRunner::ncnn_infer(const ncnn::Mat& in_blob) {
    ncnn::Extractor ex = this->net.create_extractor();
    ex.input("in0", in_blob);

    std::unordered_map<std::string, ncnn::Mat> outputs;
    for (const auto& output_name : this->net.output_names()) {
        ncnn::Mat out;
        ex.extract(output_name, out);
        outputs[output_name] = out;
    }
    return outputs;
}

std::vector<YOLOOutput> NCNNRunner::postprocess(const std::unordered_map<std::string, ncnn::Mat>& outputs, const YOLOInput& yolo_input){
    if (this->task == "cls") {
        // Classification postprocessing
        return this->clspostprocess(outputs, yolo_input);
    } else if (this->task == "det") {
        // Detection postprocessing
        return this->detpostprocess(outputs, yolo_input);
    } else if (this->task == "obb") {
        // Oriented bounding box postprocessing
        return this->obbpostprocess(outputs, yolo_input);
    } else if (this->task == "pose") {
        // Pose estimation postprocessing
        return this->posepostprocess(outputs, yolo_input);
    } else if (this->task == "seg") {
        // Segmentation postprocessing
        return this->segpostprocess(outputs, yolo_input);
    } else {
        // 仅用来屏蔽编译警告，前面已经检查过合法性了
        throw std::invalid_argument("[ERROR] Unsupported task type: " + this->task);
    }
}