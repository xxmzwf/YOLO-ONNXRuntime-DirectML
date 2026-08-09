#include "infer.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <dml_provider_factory.h>
#endif

namespace
{
#ifdef _WIN32
std::wstring widenPath(const std::string& path)
{
    if (path.empty()) {
        return {};
    }

    const int length = MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0);
    if (length <= 0) {
        return std::wstring(path.begin(), path.end());
    }

    std::wstring wide(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), wide.data(), length);
    return wide;
}
#endif

size_t elementCount(const int64_t* shape, size_t rank)
{
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) {
        count *= static_cast<size_t>(shape[i]);
    }
    return count;
}

bool isSupportedElementType(ONNXTensorElementDataType type)
{
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

#ifdef _WIN32
// a dedicated high-priority command queue keeps inference ahead of other desktop GPU work
bool appendHighPriorityDml(Ort::SessionOptions& options, int device)
{
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return false;
    }
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapters1(static_cast<UINT>(device), &adapter))) {
        return false;
    }
    ComPtr<ID3D12Device> d3dDevice;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3dDevice)))) {
        return false;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    ComPtr<ID3D12CommandQueue> queue;
    if (FAILED(d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)))) {
        return false;
    }
    ComPtr<IDMLDevice> dmlDevice;
    if (FAILED(DMLCreateDevice(d3dDevice.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)))) {
        return false;
    }
    OrtStatus* status = OrtSessionOptionsAppendExecutionProviderEx_DML(options, dmlDevice.Get(), queue.Get());
    if (status != nullptr) {
        Ort::GetApi().ReleaseStatus(status);
        return false;
    }
    return true;
}
#endif
}

struct InferEngine::Impl
{
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YoloOrtDml");
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Session session = Ort::Session(nullptr);
    Ort::RunOptions runOptions;
    Ort::IoBinding ioBinding = Ort::IoBinding(nullptr);

    std::vector<Ort::AllocatedStringPtr> nameHolders;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    ModelInputInfo info;
    bool sessionReady = false;

    Ort::Value inputValue = Ort::Value(nullptr);
    const void* inputData = nullptr;
    std::array<int64_t, 4> inputShape = {0, 0, 0, 0};

    struct OutputBuffer
    {
        std::vector<int64_t> shape;
        bool fp16 = false;
        std::vector<uint8_t> storage;
    };
    std::vector<OutputBuffer> outputBuffers;
    std::vector<Ort::Value> outputValues;
    std::vector<OutputView> views;
    bool staticOutputs = false;

    void reset()
    {
        sessionReady = false;
        ioBinding = Ort::IoBinding(nullptr);
        session = Ort::Session(nullptr);
        nameHolders.clear();
        inputNames.clear();
        outputNames.clear();
        info = ModelInputInfo();
        inputValue = Ort::Value(nullptr);
        inputData = nullptr;
        inputShape = {0, 0, 0, 0};
        outputBuffers.clear();
        outputValues.clear();
        views.clear();
        staticOutputs = false;
    }

    void createSession(const std::string& modelPath, int device)
    {
#ifdef _WIN32
        const std::wstring widePath = widenPath(modelPath);
        // the second attempt disables DML graph fusion: some fp16 YOLO exports crash the fusion pass
        for (const bool disableGraphFusion : {false, true}) {
            try {
                Ort::SessionOptions options;
                options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                options.SetIntraOpNumThreads(1);
                options.SetInterOpNumThreads(1);
                options.DisableMemPattern();
                options.SetExecutionMode(ORT_SEQUENTIAL);
                if (disableGraphFusion) {
                    options.AddConfigEntry("ep.dml.disable_graph_fusion", "1");
                }
                if (!appendHighPriorityDml(options, device)) {
                    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, device));
                }
                session = Ort::Session(env, widePath.c_str(), options);
                if (disableGraphFusion) {
                    std::cerr << "[YoloOrtDml] DirectML is running without graph fusion for this model." << std::endl;
                }
                return;
            } catch (const Ort::Exception& exception) {
                std::cerr << "[YoloOrtDml] DirectML session failed: " << exception.what() << std::endl;
            }
        }
        std::cerr << "[YoloOrtDml] DirectML is unavailable for this model; using CPU." << std::endl;
#endif
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        const unsigned int threads = std::thread::hardware_concurrency();
        options.SetIntraOpNumThreads(static_cast<int>(std::min(6u, threads == 0 ? 1u : threads)));
#ifdef _WIN32
        session = Ort::Session(env, widePath.c_str(), options);
#else
        session = Ort::Session(env, modelPath.c_str(), options);
#endif
    }

    void readModelInfo()
    {
        const size_t outputCount = session.GetOutputCount();
        if (session.GetInputCount() != 1 || outputCount == 0) {
            throw std::runtime_error("Model must have exactly one input and at least one output.");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        nameHolders.push_back(session.GetInputNameAllocated(0, allocator));
        inputNames.push_back(nameHolders.back().get());
        for (size_t i = 0; i < outputCount; ++i) {
            nameHolders.push_back(session.GetOutputNameAllocated(i, allocator));
            outputNames.push_back(nameHolders.back().get());
        }

        const Ort::TypeInfo typeInfo = session.GetInputTypeInfo(0);
        const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = tensorInfo.GetShape();
        if (shape.size() != 4) {
            throw std::runtime_error("Model input must be NCHW float or NHWC uint8.");
        }
        const ONNXTensorElementDataType type = tensorInfo.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            // baked preprocessing model: uint8 NHWC (RGB), cast/layout/normalize run in-graph
            info.rawInput = true;
            info.channels = shape[3] > 0 ? static_cast<int>(shape[3]) : 3;
            info.dynamicSize = shape[1] <= 0 || shape[2] <= 0;
            info.height = shape[1] > 0 ? static_cast<int>(shape[1]) : 640;
            info.width = shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
        } else if (isSupportedElementType(type)) {
            info.fp16 = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            info.channels = shape[1] > 0 ? static_cast<int>(shape[1]) : 3;
            info.dynamicSize = shape[2] <= 0 || shape[3] <= 0;
            info.height = shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
            info.width = shape[3] > 0 ? static_cast<int>(shape[3]) : 640;
        } else {
            throw std::runtime_error("Only float32, float16 and uint8 model inputs are supported.");
        }
        if (info.channels != 1 && info.channels != 3) {
            throw std::runtime_error("Only 1-channel and 3-channel model inputs are supported.");
        }
    }

    void prepareOutputs()
    {
        for (size_t i = 0; i < outputNames.size(); ++i) {
            const Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
            const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            OutputBuffer buffer;
            buffer.shape = tensorInfo.GetShape();
            const ONNXTensorElementDataType type = tensorInfo.GetElementType();
            const bool staticShape = std::all_of(buffer.shape.begin(), buffer.shape.end(), [](int64_t dim) {
                return dim > 0;
            });
            if (!isSupportedElementType(type) || !staticShape) {
                outputBuffers.clear();
                return;
            }

            buffer.fp16 = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            buffer.storage.resize(elementCount(buffer.shape.data(), buffer.shape.size()) * (buffer.fp16 ? 2 : 4));
            outputBuffers.push_back(std::move(buffer));
        }

        for (OutputBuffer& buffer : outputBuffers) {
            outputValues.push_back(Ort::Value::CreateTensor(
                memoryInfo,
                buffer.storage.data(),
                buffer.storage.size(),
                buffer.shape.data(),
                buffer.shape.size(),
                buffer.fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
            views.push_back({buffer.shape, buffer.fp16, buffer.storage.data()});
        }

        // pre-bind the fixed output buffers once; run() then skips per-call name resolution
        ioBinding = Ort::IoBinding(session);
        for (size_t i = 0; i < outputValues.size(); ++i) {
            ioBinding.BindOutput(outputNames[i], outputValues[i]);
        }
        staticOutputs = true;
    }

    const std::vector<OutputView>& run(InputTensor& input)
    {
        void* data;
        size_t elementBytes;
        ONNXTensorElementDataType type;
        if (input.raw) {
            data = input.byteData.data();
            elementBytes = 1;
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        } else if (input.fp16) {
            data = input.halfData.data();
            elementBytes = 2;
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        } else {
            data = input.floatData.data();
            elementBytes = 4;
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        // DirectML may retain the input resource behind an IoBinding.  The
        // preprocessing step rewrites the same CPU buffer for every frame,
        // so static-output runs must recreate and bind the current input on
        // every call instead of only binding when the address or shape changes.
        if (staticOutputs || data != inputData || input.shape != inputShape) {
            const size_t bytes = elementCount(input.shape.data(), input.shape.size()) * elementBytes;
            inputValue = Ort::Value::CreateTensor(
                memoryInfo,
                data,
                bytes,
                input.shape.data(),
                input.shape.size(),
                type);
            inputData = data;
            inputShape = input.shape;
            if (staticOutputs) {
                ioBinding.ClearBoundInputs();
                ioBinding.BindInput(inputNames[0], inputValue);
            }
        }

        if (staticOutputs) {
            session.Run(runOptions, ioBinding);
            return views;
        }

        outputValues = session.Run(runOptions, inputNames.data(), &inputValue, 1, outputNames.data(), outputNames.size());
        views.clear();
        for (const Ort::Value& value : outputValues) {
            const auto tensorInfo = value.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType type = tensorInfo.GetElementType();
            if (!isSupportedElementType(type)) {
                continue;
            }
            views.push_back({tensorInfo.GetShape(), type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, value.GetTensorRawData()});
        }
        return views;
    }
};

InferEngine::InferEngine()
    : impl(std::make_unique<Impl>())
{
}

InferEngine::~InferEngine() = default;

bool InferEngine::loadModel(const std::string& modelPath, int device)
{
    impl->reset();
    try {
        impl->createSession(modelPath, device);
        impl->readModelInfo();
        impl->prepareOutputs();
        impl->sessionReady = true;
        return true;
    } catch (const std::exception& exception) {
        std::cerr << "[YoloOrtDml] failed to load model: " << exception.what() << std::endl;
        impl->reset();
        return false;
    }
}

bool InferEngine::ready() const
{
    return impl->sessionReady;
}

const ModelInputInfo& InferEngine::inputInfo() const
{
    return impl->info;
}

const std::vector<OutputView>& InferEngine::run(InputTensor& input)
{
    return impl->run(input);
}
