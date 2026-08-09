#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define YOLOORTDML_SSE 1
#endif

namespace
{
float clampf(float value, float low, float high)
{
    return value < low ? low : (value > high ? high : value);
}

void addBox(std::vector<DetectResultBox>& boxes, const PreprocessResult& preprocess,
            float left, float top, float right, float bottom, float score, int classId)
{
    left = clampf((left - preprocess.padX) * preprocess.invScale, 0.0f, static_cast<float>(preprocess.imageWidth));
    top = clampf((top - preprocess.padY) * preprocess.invScale, 0.0f, static_cast<float>(preprocess.imageHeight));
    right = clampf((right - preprocess.padX) * preprocess.invScale, 0.0f, static_cast<float>(preprocess.imageWidth));
    bottom = clampf((bottom - preprocess.padY) * preprocess.invScale, 0.0f, static_cast<float>(preprocess.imageHeight));
    const float width = right - left;
    const float height = bottom - top;
    if (width <= 0.0f || height <= 0.0f || !std::isfinite(score)) {
        return;
    }
    boxes.push_back({left, top, width, height, score, classId});
}

void addCenterBox(std::vector<DetectResultBox>& boxes, const PreprocessResult& preprocess,
                  float centerX, float centerY, float width, float height, float score, int classId)
{
    addBox(boxes, preprocess,
           centerX - 0.5f * width, centerY - 0.5f * height,
           centerX + 0.5f * width, centerY + 0.5f * height,
           score, classId);
}

float maxScoreFloat(const float* scores, int count)
{
    int i = 0;
    float best = scores[0];
#ifdef YOLOORTDML_SSE
    if (count >= 8) {
        __m128 vbest = _mm_loadu_ps(scores);
        for (i = 4; i + 4 <= count; i += 4) {
            vbest = _mm_max_ps(vbest, _mm_loadu_ps(scores + i));
        }
        vbest = _mm_max_ps(vbest, _mm_movehl_ps(vbest, vbest));
        vbest = _mm_max_ps(vbest, _mm_shuffle_ps(vbest, vbest, 0x55));
        best = _mm_cvtss_f32(vbest);
    }
#endif
    for (; i < count; ++i) {
        best = std::max(best, scores[i]);
    }
    return best;
}

int findClassFloat(const float* scores, int count, float value)
{
    int classId = 0;
    while (classId + 1 < count && scores[classId] != value) {
        ++classId;
    }
    return classId;
}

uint16_t maxScoreHalf(const uint16_t* scores, int count)
{
    int i = 0;
    uint16_t best = scores[0];
#ifdef YOLOORTDML_SSE
    // non-negative half bit patterns order like their values, also under signed 16-bit compare
    if (count >= 16) {
        __m128i vbest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scores));
        for (i = 8; i + 8 <= count; i += 8) {
            vbest = _mm_max_epi16(vbest, _mm_loadu_si128(reinterpret_cast<const __m128i*>(scores + i)));
        }
        vbest = _mm_max_epi16(vbest, _mm_srli_si128(vbest, 8));
        vbest = _mm_max_epi16(vbest, _mm_srli_si128(vbest, 4));
        vbest = _mm_max_epi16(vbest, _mm_srli_si128(vbest, 2));
        best = static_cast<uint16_t>(_mm_extract_epi16(vbest, 0));
    }
#endif
    for (; i < count; ++i) {
        best = std::max(best, scores[i]);
    }
    return best;
}

int findClassHalf(const uint16_t* scores, int count, uint16_t bits)
{
    int classId = 0;
    while (classId + 1 < count && scores[classId] != bits) {
        ++classId;
    }
    return classId;
}

// boxes: [1, N, 4 + (objectness) + classes], one detection per row
void collectRowMajor(const OutputView& output, const PreprocessResult& preprocess,
                     float confidence, bool objectness, int classes, std::vector<DetectResultBox>& boxes)
{
    const size_t boxCount = static_cast<size_t>(output.shape[1]);
    const size_t features = static_cast<size_t>(output.shape[2]);
    const size_t classOffset = objectness ? 5 : 4;

    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        for (size_t i = 0; i < boxCount; ++i) {
            const float* row = values + i * features;
            const float objectScore = objectness ? row[4] : 1.0f;
            if (objectScore <= confidence) {
                continue;
            }
            const float* scores = row + classOffset;
            const float bestScore = maxScoreFloat(scores, classes);
            const float score = objectScore * bestScore;
            if (score <= confidence) {
                continue;
            }
            const int classId = findClassFloat(scores, classes, bestScore);
            addCenterBox(boxes, preprocess, row[0], row[1], row[2], row[3], score, classId);
        }
        return;
    }

    // non-negative half bit patterns compare like their float values
    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t confidenceBits = floatToHalfBits(confidence);
    for (size_t i = 0; i < boxCount; ++i) {
        const uint16_t* row = values + i * features;
        if (objectness && row[4] <= confidenceBits) {
            continue;
        }
        const uint16_t* scores = row + classOffset;
        const uint16_t bestBits = maxScoreHalf(scores, classes);
        float score = halfBitsToFloat(bestBits);
        if (objectness) {
            score *= halfBitsToFloat(row[4]);
        }
        if (score <= confidence) {
            continue;
        }
        const int classId = findClassHalf(scores, classes, bestBits);
        addCenterBox(boxes, preprocess,
                     halfBitsToFloat(row[0]), halfBitsToFloat(row[1]),
                     halfBitsToFloat(row[2]), halfBitsToFloat(row[3]),
                     score, classId);
    }
}

void planarMaxFloat(const float* classData, int classes, size_t count, std::vector<float>& best)
{
    best.assign(classData, classData + count);
    float* dst = best.data();
    for (int c = 1; c < classes; ++c) {
        const float* scores = classData + static_cast<size_t>(c) * count;
        size_t i = 0;
#ifdef YOLOORTDML_SSE
        for (; i + 4 <= count; i += 4) {
            _mm_storeu_ps(dst + i, _mm_max_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(scores + i)));
        }
#endif
        for (; i < count; ++i) {
            dst[i] = std::max(dst[i], scores[i]);
        }
    }
}

void planarMaxHalf(const uint16_t* classData, int classes, size_t count, std::vector<uint16_t>& best)
{
    best.assign(classData, classData + count);
    uint16_t* dst = best.data();
    for (int c = 1; c < classes; ++c) {
        const uint16_t* scores = classData + static_cast<size_t>(c) * count;
        size_t i = 0;
#ifdef YOLOORTDML_SSE
        // scores are non-negative halves (< 0x8000), so signed 16-bit max matches float order
        for (; i + 8 <= count; i += 8) {
            const __m128i bestBits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
            const __m128i scoreBits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scores + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_max_epi16(bestBits, scoreBits));
        }
#endif
        for (; i < count; ++i) {
            dst[i] = std::max(dst[i], scores[i]);
        }
    }
}

// boxes: [1, 4 + (objectness) + classes, N], one feature plane per row
void collectPlanar(const OutputView& output, const PreprocessResult& preprocess,
                   float confidence, bool objectness, int classes, PostprocessContext& context)
{
    const size_t boxCount = static_cast<size_t>(output.shape[2]);
    const size_t classOffset = objectness ? 5 : 4;
    std::vector<DetectResultBox>& boxes = context.candidates;

    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        const float* xData = values;
        const float* yData = values + boxCount;
        const float* wData = values + 2 * boxCount;
        const float* hData = values + 3 * boxCount;
        const float* classData = values + classOffset * boxCount;

        if (!objectness) {
            planarMaxFloat(classData, classes, boxCount, context.bestScores);
            for (size_t i = 0; i < boxCount; ++i) {
                const float score = context.bestScores[i];
                if (score <= confidence) {
                    continue;
                }
                int classId = 0;
                while (classId + 1 < classes && classData[static_cast<size_t>(classId) * boxCount + i] != score) {
                    ++classId;
                }
                addCenterBox(boxes, preprocess, xData[i], yData[i], wData[i], hData[i], score, classId);
            }
            return;
        }

        const float* objectData = values + 4 * boxCount;
        planarMaxFloat(classData, classes, boxCount, context.bestScores);
        for (size_t i = 0; i < boxCount; ++i) {
            const float objectScore = objectData[i];
            if (objectScore <= confidence) {
                continue;
            }
            const float bestScore = context.bestScores[i];
            const float score = objectScore * bestScore;
            if (score <= confidence) {
                continue;
            }
            int classId = 0;
            while (classId + 1 < classes && classData[static_cast<size_t>(classId) * boxCount + i] != bestScore) {
                ++classId;
            }
            addCenterBox(boxes, preprocess, xData[i], yData[i], wData[i], hData[i], score, classId);
        }
        return;
    }

    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t* xData = values;
    const uint16_t* yData = values + boxCount;
    const uint16_t* wData = values + 2 * boxCount;
    const uint16_t* hData = values + 3 * boxCount;
    const uint16_t* classData = values + classOffset * boxCount;
    const uint16_t confidenceBits = floatToHalfBits(confidence);

    if (!objectness) {
        planarMaxHalf(classData, classes, boxCount, context.bestHalfScores);
        for (size_t i = 0; i < boxCount; ++i) {
            const uint16_t scoreBits = context.bestHalfScores[i];
            if (scoreBits <= confidenceBits) {
                continue;
            }
            int classId = 0;
            while (classId + 1 < classes && classData[static_cast<size_t>(classId) * boxCount + i] != scoreBits) {
                ++classId;
            }
            addCenterBox(boxes, preprocess,
                         halfBitsToFloat(xData[i]), halfBitsToFloat(yData[i]),
                         halfBitsToFloat(wData[i]), halfBitsToFloat(hData[i]),
                         halfBitsToFloat(scoreBits), classId);
        }
        return;
    }

    const uint16_t* objectData = values + 4 * boxCount;
    planarMaxHalf(classData, classes, boxCount, context.bestHalfScores);
    for (size_t i = 0; i < boxCount; ++i) {
        if (objectData[i] <= confidenceBits) {
            continue;
        }
        const uint16_t bestBits = context.bestHalfScores[i];
        const float score = halfBitsToFloat(objectData[i]) * halfBitsToFloat(bestBits);
        if (score <= confidence) {
            continue;
        }
        int classId = 0;
        while (classId + 1 < classes && classData[static_cast<size_t>(classId) * boxCount + i] != bestBits) {
            ++classId;
        }
        addCenterBox(boxes, preprocess,
                     halfBitsToFloat(xData[i]), halfBitsToFloat(yData[i]),
                     halfBitsToFloat(wData[i]), halfBitsToFloat(hData[i]),
                     score, classId);
    }
}

// boxes: [1, N, 6] rows of [x1, y1, x2, y2, score, classId], NMS already inside the model
void collectEndToEnd(const OutputView& output, const PreprocessResult& preprocess,
                     float confidence, std::vector<DetectResultBox>& boxes)
{
    const size_t boxCount = static_cast<size_t>(output.shape[1]);

    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        for (size_t i = 0; i < boxCount; ++i) {
            const float* row = values + i * 6;
            if (row[4] <= confidence) {
                continue;
            }
            addBox(boxes, preprocess, row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5]));
        }
        return;
    }

    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t confidenceBits = floatToHalfBits(confidence);
    for (size_t i = 0; i < boxCount; ++i) {
        const uint16_t* row = values + i * 6;
        if (row[4] <= confidenceBits) {
            continue;
        }
        addBox(boxes, preprocess,
               halfBitsToFloat(row[0]), halfBitsToFloat(row[1]),
               halfBitsToFloat(row[2]), halfBitsToFloat(row[3]),
               halfBitsToFloat(row[4]), static_cast<int>(halfBitsToFloat(row[5])));
    }
}

float intersectionOverUnion(const DetectResultBox& lhs, const DetectResultBox& rhs)
{
    const float x1 = std::max(lhs.x, rhs.x);
    const float y1 = std::max(lhs.y, rhs.y);
    const float x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const float width = x2 - x1;
    const float height = y2 - y1;
    if (width <= 0.0f || height <= 0.0f) {
        return 0.0f;
    }
    const float intersection = width * height;
    return intersection / (lhs.width * lhs.height + rhs.width * rhs.height - intersection);
}

void applyNms(PostprocessContext& context, float nmsThreshold, std::vector<DetectResultBox>& results)
{
    const std::vector<DetectResultBox>& candidates = context.candidates;
    if (candidates.empty()) {
        return;
    }

    context.order.resize(candidates.size());
    std::iota(context.order.begin(), context.order.end(), 0);
    std::sort(context.order.begin(), context.order.end(), [&candidates](int lhs, int rhs) {
        return candidates[lhs].score > candidates[rhs].score;
    });
    context.suppressed.assign(candidates.size(), 0);

    results.reserve(candidates.size());
    for (size_t i = 0; i < context.order.size(); ++i) {
        const int current = context.order[i];
        if (context.suppressed[current]) {
            continue;
        }
        const DetectResultBox& keep = candidates[current];
        results.push_back(keep);

        for (size_t j = i + 1; j < context.order.size(); ++j) {
            const int other = context.order[j];
            if (context.suppressed[other]) {
                continue;
            }
            const DetectResultBox& box = candidates[other];
            if (box.classId == keep.classId && intersectionOverUnion(keep, box) > nmsThreshold) {
                context.suppressed[other] = 1;
            }
        }
    }
}
}

void postprocess(
    const std::vector<OutputView>& outputs,
    const PreprocessResult& preprocess,
    float confidenceThreshold,
    float nmsThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& results)
{
    results.clear();
    context.candidates.clear();
    if (outputs.empty() || preprocess.imageWidth <= 0 || preprocess.imageHeight <= 0) {
        return;
    }
    const OutputView& output = outputs[0];
    if (output.data == nullptr || output.shape.size() != 3 || output.shape[1] <= 0 || output.shape[2] <= 4) {
        return;
    }

    const int64_t rows = output.shape[1];
    const int64_t cols = output.shape[2];
    if (cols == 6) {
        collectEndToEnd(output, preprocess, confidenceThreshold, context.candidates);
        applyNms(context, nmsThreshold, results);
        return;
    }

    const bool rowMajor = rows > cols;
    const size_t features = static_cast<size_t>(rowMajor ? cols : rows);
    if (features <= 4) {
        return;
    }
    // Supported YOLO layouts use objectness in row-major outputs and omit it in planar outputs.
    const bool objectness = rowMajor;
    const int classes = static_cast<int>(features - (objectness ? 5 : 4));

    if (rowMajor) {
        collectRowMajor(output, preprocess, confidenceThreshold, objectness, classes, context.candidates);
    } else {
        collectPlanar(output, preprocess, confidenceThreshold, objectness, classes, context);
    }
    applyNms(context, nmsThreshold, results);
}
