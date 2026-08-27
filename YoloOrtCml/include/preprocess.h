#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <vector>

#include "YoloOrtCml.h"
#include "infer.h"

struct PreprocessResult
{
    int imageWidth = 0;
    int imageHeight = 0;
    float invScale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

struct PreprocessContext
{
    std::vector<int> xOffset0;
    std::vector<int> xOffset1;
    std::vector<float> xWeight;
    std::vector<float> rowA;
    std::vector<float> rowB;
    int xSafe = 0;    // pixels before this index allow 4-byte SIMD loads in 3-byte formats
};

PreprocessResult preprocess(
    const ModelInputInfo& info,
    const ImageView& image,
    PreprocessContext& context,
    InputTensor& tensor);

#endif // PREPROCESS_H
