#ifndef POSTPROCESS_H
#define POSTPROCESS_H
#include <cstdint>
#include <vector>

#include "YoloOrtDml.h"
#include "infer.h"
#include "preprocess.h"

struct PostprocessContext
{
    std::vector<DetectResultBox> candidates;
    std::vector<float> bestScores;
    std::vector<uint16_t> bestHalfScores;
    std::vector<int> order;
    std::vector<unsigned char> suppressed;
};

void postprocess(
    const std::vector<OutputView>& outputs,
    const PreprocessResult& preprocess,
    float confidenceThreshold,
    float nmsThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& results);

#endif // POSTPROCESS_H
