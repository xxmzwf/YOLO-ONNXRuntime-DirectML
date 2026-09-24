#include "YoloOrtDml.h"

#include <iostream>
#include <utility>

#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

namespace YOD
{
struct YoloOrtDml::Impl
{
    InferEngine engine;
    std::string modelPath;
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
    impl->preprocessResult = YOD::preprocess(impl->engine.inputInfo(), impl->image, impl->preprocessContext, impl->inputTensor);
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
    YOD::postprocess(
        *impl->outputs,
        impl->preprocessResult,
        impl->confidenceThreshold,
        impl->nmsThreshold,
        impl->postprocessContext,
        impl->results);
}

std::vector<DetectResultBox> YoloOrtDml::resultBoxes()
{
    return impl->results;
}
} // namespace YOD
