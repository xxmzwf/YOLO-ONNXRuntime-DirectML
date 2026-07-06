#ifndef INFER_H
#define INFER_H
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct ModelInputInfo
{
    int width = 640;
    int height = 640;
    int channels = 3;
    bool fp16 = false;
    bool rawInput = false;    // uint8 NHWC input, preprocessing baked into the model
    bool dynamicSize = false;
};

struct InputTensor
{
    std::array<int64_t, 4> shape = {1, 3, 0, 0};
    bool fp16 = false;
    bool raw = false;
    std::vector<float> floatData;
    std::vector<uint16_t> halfData;
    std::vector<uint8_t> byteData;
};

struct OutputView
{
    std::vector<int64_t> shape;
    bool fp16 = false;
    const void* data = nullptr;
};

// IEEE 754 half <-> float for tensor data; values below the half normal range flush to zero.
inline uint16_t floatToHalfBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    bits &= 0x7FFFFFFFu;
    if (bits >= 0x477FF000u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (bits < 0x38800000u) {
        return static_cast<uint16_t>(sign);
    }
    bits += 0x00001000u;
    return static_cast<uint16_t>(sign | ((bits - 0x38000000u) >> 13));
}

inline float halfBitsToFloat(uint16_t half)
{
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t magnitude = half & 0x7FFFu;
    const uint32_t bits = magnitude >= 0x0400u ? sign | ((magnitude + 0x1C000u) << 13) : sign;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

class InferEngine
{
public:
    InferEngine();
    ~InferEngine();

    bool loadModel(const std::string& modelPath, int device);
    bool ready() const;
    const ModelInputInfo& inputInfo() const;
    const std::vector<OutputView>& run(InputTensor& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // INFER_H