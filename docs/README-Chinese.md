# YoloOrtDml

**Windows 平台高速 YOLO 目标检测 —— ONNX Runtime + DirectML，零 OpenCV 依赖。**

一个以"最低单帧延迟"为目标打磨的精简 C++ 推理库：YOLOv6-N @ 320 在笔记本 RTX 3060 上，
完整检测流程（预处理 → 推理 → 后处理）**不到 1 毫秒**。

[English](../README.md) | 中文文档

![推理结果](../assets/result.png)

*使用 `yolov6n_320_fp16_u8.onnx` 的检测效果与分阶段耗时。*

## 亮点

- **端到端约 1100 FPS**（YOLOv6-N 320，RTX 3060 Laptop）—— 见[性能](#性能)
- **DirectML GPU 推理**：任何支持 DirectX 12 的 GPU 都能跑（NVIDIA / AMD / Intel），不需要 CUDA 或任何厂商 SDK
- **不依赖 OpenCV**：本库只依赖 ONNX Runtime；预处理为手写 SIMD（SSSE3 通道解交织、F16C 半精度转换、缩放+归一化+letterbox 单遍融合）
- **热路径零分配**：IoBinding 预绑定、输入输出张量预分配、张量对象缓存——首帧之后不再有任何分配或名字解析
- **u8 烧制模型**（[tools/bake_preprocess.py](../tools/bake_preprocess.py)）：布局转换、归一化和 fp16 输出转换被移入 ONNX 图内由 GPU 执行；CPU 预处理退化为一次行拷贝（约 0.02 ms），PCIe 传输量降为 1/4
- **模型自动适配**：输入分辨率（256 / 320 / 640 / ...）、fp32 / fp16 / uint8 输入、输出布局全部从模型元数据自动识别——换模型不需要改任何代码
- 自建高优先级 D3D12 命令队列，GPU 被其他程序占用时推理仍能优先执行

## 支持的模型

| 系列 | 输出布局 | 备注 |
|---|---|---|
| YOLOv5 / v6 / v7 风格 | 行主序 `[1, N, 4+1+C]`（带 objectness） | |
| YOLOv8 / v9 / v11 / v12 风格 | 平面 `[1, 4+C, N]` | |
| YOLOv10 / v26 端到端 | `[1, N, 6]` | NMS 在模型内部 |

- fp32 或 fp16 权重；float32、float16 或烧制后的 uint8 输入张量
- 任意静态输入分辨率、1 或 3 通道输入，均从模型中读取
- 提供标签文件可对自定义类别数做精确布局判定；不提供时使用形状启发式

> 注意：yolov10 / yolo26 的 **fp16** 导出目前会触发 ONNX Runtime 中 DirectML 图融合的缺陷
>（上游问题）导致回退降速——在 DirectML 上请使用它们的 **fp32** 导出。

## 性能

测试条件：320×320 BGR 输入（与模型尺寸一致），预热后连续 100 帧取平均。
硬件：Intel i7-12700H + NVIDIA GeForce RTX 3060 Laptop GPU，Windows 11，ONNX Runtime 1.28（DirectML）。

| 模型 | 预处理 | 推理 | 后处理 | 总计 | FPS |
|---|---|---|---|---|---|
| `yolov5n_320_fp16_u8.onnx` | 0.022 ms | 1.34 ms | 0.013 ms | **1.38 ms** | ~725 |
| `yolov6n_320_fp16_u8.onnx` | 0.019 ms | 0.86 ms | 0.013 ms | **0.89 ms** | ~1120 |

当输入图片尺寸与模型尺寸不一致时，SIMD 缩放路径约增加 0.14–0.19 ms。

## 环境依赖

- Windows 10 / 11，支持 DirectX 12 的 GPU
- **带 DirectML 执行提供器的 ONNX Runtime 动态库**（1.28 测试通过）—— 提供 `onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll`
- MSVC 2022（C++17），CMake ≥ 3.16
- **不需要 OpenCV** —— 本库完全不使用它；下面示例中的 OpenCV 仅用于读图和显示
- Python 3 并 `pip install onnx onnxruntime` —— 仅在使用可选的模型工具时需要

## 编译与安装

1. 打开 [CMakeLists.txt](../CMakeLists.txt)，把 `ONNXRUNTIME_PATH` 改成你电脑上 ONNX Runtime DirectML 包的实际路径：

```cmake
set(ONNXRUNTIME_PATH "D:/CodeLibraries/ONNXRuntime-1.28.0-Shared")   # <-- 改成你的路径
```

该包应包含 `include/onnxruntime`、`lib/onnxruntime.lib`、`lib/cmake/onnxruntime` 和 `bin/*.dll`。

2. 配置、编译、安装：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix D:/libs/YoloOrtDml
```

安装目录包含使用方需要的全部内容：`YoloOrtDml.dll` 与导入库、唯一的公开头文件 `YoloOrtDml.h`、
ONNX Runtime / DirectML 运行时 DLL，以及 CMake 包配置文件。

## 在你的项目中使用本库

```cmake
list(APPEND CMAKE_PREFIX_PATH "D:/libs/YoloOrtDml")    # 你的安装路径
find_package(YoloOrtDml CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE YoloOrtDml::YoloOrtDml)

# 把所有运行时 DLL 拷贝到可执行文件旁边
add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${YoloOrtDml_RUNTIME_DLLS}
        "$<TARGET_FILE_DIR:my_app>"
    VERBATIM
)
```

`YoloOrtDml_RUNTIME_DLLS` 由包配置提供，包含 `YoloOrtDml.dll`、`onnxruntime.dll`、
`onnxruntime_providers_shared.dll` 和 `DirectML.dll`。

## 模型工具

[tools/](../tools) 下的两个可选 Python 脚本能进一步榨出速度（先 `pip install onnx onnxruntime`）。

### bake_preprocess.py —— 把预处理搬到 GPU

```bash
python tools/bake_preprocess.py yolov6n_320.onnx        # -> yolov6n_320_u8.onnx
```

把模型输入改写为 `uint8[1,H,W,3]`（NHWC、RGB），并在图前端插入 `Transpose → Cast → Mul(1/255)`
节点，让布局转换与归一化在 GPU 上、模型内部完成。当网络权重为 fp16 时，还会把 fp32 输出
转成 fp16（无损——回读量减半）。输入分辨率从模型中读取，256 / 320 / 640 的导出都能烧。

引擎会自动识别烧制模型（uint8 输入），API 无需任何改动。效果：CPU 预处理变成纯行拷贝，
上传字节量相比 float32 张量下降 4 倍。

### convert_fp16.py —— 把 fp32 模型转成 fp16

```bash
python tools/convert_fp16.py yolov6n_320_fp32.onnx      # -> yolov6n_320_fp32_fp16.onnx
```

把权重和输入输出都转成 fp16（上传与回读同时减半）；fp16 下不安全的算子（NMS / TopK /
Resize 等）会自动保留 fp32。权重已是 fp16 的模型会被检测并跳过。普通模型和已烧制的
u8 模型都能转。

对全新的 fp32 导出，**推荐处理链**：

```bash
python tools/convert_fp16.py  model_fp32.onnx           # fp16 权重 + fp16 输入输出
python tools/bake_preprocess.py model_fp32_fp16.onnx    # + u8 输入、GPU 预处理
```

## API

对外只暴露一个头文件 [YoloOrtDml.h](../YoloOrtDml/include/YoloOrtDml.h)——不泄漏任何 ONNX Runtime 类型。

| 方法 | 说明 |
|---|---|
| `bool setModel(std::string modelPath)` | 加载 ONNX 模型并创建 DirectML 会话（失败回退 CPU）。输入尺寸/类型/布局自动识别。 |
| `bool setLabel(std::string labelPath)` | 加载类别名（每行一个）。让输出布局判定变为精确匹配；`classId` 即此列表的下标。 |
| `void setDevice(int device)` | GPU 适配器序号（DXGI 枚举顺序），默认 0。若模型已加载会自动重建会话。 |
| `void setConfidenceThreshold(float)` | 置信度阈值。 |
| `void setNMSThreshold(float)` | NMS 的 IoU 阈值。 |
| `void setImage(ImageView&)` | 保存图像的**非拥有**视图。像素数据须保持有效直到 `preprocess()` 返回。 |
| `void preprocess()` | 图像 → 输入张量（SIMD 缩放/转换；烧制模型则为行拷贝）。 |
| `void infer()` | 通过预绑定的 IoBinding 在 GPU 上执行一次 `Run`。 |
| `void postprocess()` | 解码 + NMS。 |
| `std::vector<DetectResultBox> resultBoxes()` | 原图像素坐标系下的检测框：`x, y, width, height, score, classId`。 |

三个流水线阶段刻意拆开，方便分别统计耗时。
`ImageView` 支持 `BGR8`、`RGB8`、`BGRA8`、`RGBA8`、`GRAY8`，行跨度（stride）任意。

## API 示例

示例中的 OpenCV **仅**用于读图和显示——本库自身不依赖 OpenCV。

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>

#include "YoloOrtDml.h"

int main()
{
    // -------------------- 初始化配置 --------------------
    YoloOrtDml detector;
    detector.setDevice(0);                    // GPU 适配器序号
    detector.setConfidenceThreshold(0.3f);
    detector.setNMSThreshold(0.45f);
    if (!detector.setLabel("label.txt")) {
        std::cerr << "标签加载失败" << std::endl;
        return 1;
    }
    if (!detector.setModel("yolov6n_320_fp16_u8.onnx")) {
        std::cerr << "模型加载失败" << std::endl;
        return 1;
    }

    // -------------------- 封装 cv::Mat（不复制像素） --------------------
    cv::Mat image = cv::imread("test.jpg");   // 8 位 BGR
    ImageView view;
    view.data = image.data;                   // 像素数据指针
    view.width = image.cols;
    view.height = image.rows;
    view.channels = image.channels();
    view.stride = image.step;                 // 每行字节数（兼容带填充的 stride）
    view.format = ImageFormat::BGR8;          // BGR8 / RGB8 / BGRA8 / RGBA8 / GRAY8

    // -------------------- 执行推理 --------------------
    detector.setImage(view);                  // 像素须保持有效直到 preprocess() 返回
    detector.preprocess();
    detector.infer();
    detector.postprocess();

    // -------------------- 结果处理 --------------------
    for (const DetectResultBox& box : detector.resultBoxes()) {
        std::cout << "class " << box.classId << " score " << box.score
                  << " box [" << box.x << ", " << box.y << ", "
                  << box.width << ", " << box.height << "]" << std::endl;
        cv::rectangle(image,
                      cv::Rect(cvRound(box.x), cvRound(box.y),
                               cvRound(box.width), cvRound(box.height)),
                      cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("YoloOrtDml", image);
    cv::waitKey(0);
    return 0;
}
```

部署：把 `YoloOrtDml.dll`、`onnxruntime.dll`、`onnxruntime_providers_shared.dll`、
`DirectML.dll` 放到你的可执行文件旁边（上面的 CMake 片段会自动完成）。
