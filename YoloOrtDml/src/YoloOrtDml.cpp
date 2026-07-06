#include "YoloOrtDml.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

struct YoloOrtDml::Impl
{
    InferEngine engine;
    std::string modelPath;
    std::vector<std::string> labels;
    int device = 0;
    float confidenceThreshold = 0.4f;
    float nmsThreshold = 0.45f;

    ImageView image;
    PreprocessContext preprocessContext;
    PreprocessResult preprocessResult;
    InputTensor inputTensor;
    const std::vector<OutputView>* outputs = nullptr;
    PostprocessContext postprocessContext;
    std::vector<DetectResultBox> results;
};

YoloOrtDml::YoloOrtDml()
    : impl(std::make_unique<Impl>())
{
}

YoloOrtDml::~YoloOrtDml() = default;

bool YoloOrtDml::setModel(std::string modelPath)
{
    impl->modelPath = std::move(modelPath);
    impl->outputs = nullptr;
    return impl->engine.loadModel(impl->modelPath, impl->device);
}

bool YoloOrtDml::setLabel(std::string labelPath)
{
    std::ifstream file(std::filesystem::u8path(labelPath));
    if (!file.is_open()) {
        return false;
    }

    impl->labels.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("\xEF\xBB\xBF", 0) == 0) {
            line.erase(0, 3);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            impl->labels.push_back(line);
        }
    }
    return !impl->labels.empty();
}

void YoloOrtDml::setDevice(int device)
{
    if (device == impl->device) {
        return;
    }
    impl->device = device;
    if (impl->engine.ready()) {
        impl->outputs = nullptr;
        impl->engine.loadModel(impl->modelPath, device);
    }
}

void YoloOrtDml::setConfidenceThreshold(float threshold)
{
    impl->confidenceThreshold = threshold;
}

void YoloOrtDml::setNMSThreshold(float threshold)
{
    impl->nmsThreshold = threshold;
}

void YoloOrtDml::setImage(ImageView& image)
{
    impl->image = image;
    impl->outputs = nullptr;
}

void YoloOrtDml::preprocess()
{
    if (!impl->engine.ready() || impl->image.data == nullptr || impl->image.width <= 0 || impl->image.height <= 0) {
        return;
    }
    impl->preprocessResult = ::preprocess(impl->engine.inputInfo(), impl->image, impl->preprocessContext, impl->inputTensor);
}

void YoloOrtDml::infer()
{
    impl->outputs = nullptr;
    if (!impl->engine.ready() || (impl->inputTensor.floatData.empty() && impl->inputTensor.byteData.empty())) {
        return;
    }
    try {
        impl->outputs = &impl->engine.run(impl->inputTensor);
    } catch (const std::exception& exception) {
        std::cerr << "[YoloOrtDml] inference failed: " << exception.what() << std::endl;
    }
}

void YoloOrtDml::postprocess()
{
    impl->results.clear();
    if (impl->outputs == nullptr) {
        return;
    }
    ::postprocess(
        *impl->outputs,
        impl->preprocessResult,
        impl->confidenceThreshold,
        impl->nmsThreshold,
        static_cast<int>(impl->labels.size()),
        impl->postprocessContext,
        impl->results);
}

std::vector<DetectResultBox> YoloOrtDml::resultBoxes()
{
    return impl->results;
}