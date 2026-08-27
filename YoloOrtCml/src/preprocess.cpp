#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define YOLOORTCML_SSE 1
#if defined(_MSC_VER) || defined(__SSSE3__)
#define YOLOORTCML_SSSE3 1
#endif
#if defined(_MSC_VER) || defined(__F16C__)
#define YOLOORTCML_F16C 1
#endif
#endif

namespace
{
struct PixelLayout
{
    int step;
    int r;
    int g;
    int b;
};

PixelLayout pixelLayout(ImageFormat format)
{
    switch (format) {
    case ImageFormat::RGB8:
        return {3, 0, 1, 2};
    case ImageFormat::BGRA8:
        return {4, 2, 1, 0};
    case ImageFormat::RGBA8:
        return {4, 0, 1, 2};
    case ImageFormat::GRAY8:
        return {1, 0, 0, 0};
    case ImageFormat::BGR8:
    default:
        return {3, 2, 1, 0};
    }
}

int alignUp(int value, int alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void buildXMap(int srcWidth, int dstWidth, int step, PreprocessContext& context)
{
    context.xOffset0.resize(dstWidth);
    context.xOffset1.resize(dstWidth);
    context.xWeight.resize(dstWidth);
    const float ratio = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    for (int x = 0; x < dstWidth; ++x) {
        const float srcX = (static_cast<float>(x) + 0.5f) * ratio - 0.5f;
        int x0 = static_cast<int>(std::floor(srcX));
        float weight = srcX - static_cast<float>(x0);
        if (x0 < 0) {
            x0 = 0;
            weight = 0.0f;
        }
        context.xOffset0[x] = x0 * step;
        context.xOffset1[x] = std::min(x0 + 1, srcWidth - 1) * step;
        context.xWeight[x] = weight;
    }

    context.xSafe = dstWidth;
    if (step == 3) {
        const int limit = (srcWidth - 1) * 3;
        while (context.xSafe > 0 && context.xOffset1[context.xSafe - 1] >= limit) {
            --context.xSafe;
        }
    }
}

void hresizeRgb(const uint8_t* src, const PixelLayout& layout, const PreprocessContext& context,
                int width, float* r, float* g, float* b)
{
    const int* offset0 = context.xOffset0.data();
    const int* offset1 = context.xOffset1.data();
    const float* weights = context.xWeight.data();
    const int ri = layout.r;
    const int gi = layout.g;
    const int bi = layout.b;
    for (int x = 0; x < width; ++x) {
        const uint8_t* p0 = src + offset0[x];
        const uint8_t* p1 = src + offset1[x];
        const float w = weights[x];
        r[x] = static_cast<float>(p0[ri]) + (static_cast<float>(p1[ri]) - static_cast<float>(p0[ri])) * w;
        g[x] = static_cast<float>(p0[gi]) + (static_cast<float>(p1[gi]) - static_cast<float>(p0[gi])) * w;
        b[x] = static_cast<float>(p0[bi]) + (static_cast<float>(p1[bi]) - static_cast<float>(p0[bi])) * w;
    }
}

// GRAY8 maps r/g/b to the same byte, so the luma weights sum back to the original value.
void hresizeLuma(const uint8_t* src, const PixelLayout& layout, const PreprocessContext& context,
                 int width, float* dst)
{
    const int* offset0 = context.xOffset0.data();
    const int* offset1 = context.xOffset1.data();
    const float* weights = context.xWeight.data();
    const int ri = layout.r;
    const int gi = layout.g;
    const int bi = layout.b;
    for (int x = 0; x < width; ++x) {
        const uint8_t* p0 = src + offset0[x];
        const uint8_t* p1 = src + offset1[x];
        const float luma0 = 0.299f * p0[ri] + 0.587f * p0[gi] + 0.114f * p0[bi];
        const float luma1 = 0.299f * p1[ri] + 0.587f * p1[gi] + 0.114f * p1[bi];
        dst[x] = luma0 + (luma1 - luma0) * weights[x];
    }
}

void vresizeStore(const float* row0, const float* row1, float weight, int width, float* dst)
{
    constexpr float normalize = 1.0f / 255.0f;
    int x = 0;
#ifdef YOLOORTCML_SSE
    const __m128 w = _mm_set1_ps(weight);
    const __m128 scale = _mm_set1_ps(normalize);
    for (; x + 4 <= width; x += 4) {
        const __m128 a = _mm_loadu_ps(row0 + x);
        const __m128 b = _mm_loadu_ps(row1 + x);
        _mm_storeu_ps(dst + x, _mm_mul_ps(_mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), w)), scale));
    }
#endif
    for (; x < width; ++x) {
        dst[x] = (row0[x] + (row1[x] - row0[x]) * weight) * normalize;
    }
}

#ifdef YOLOORTCML_SSE
void storeU8AsFloat(__m128i bytes, __m128 scale, float* dst)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i lo = _mm_unpacklo_epi8(bytes, zero);
    const __m128i hi = _mm_unpackhi_epi8(bytes, zero);
    _mm_storeu_ps(dst, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(lo, zero)), scale));
    _mm_storeu_ps(dst + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, zero)), scale));
    _mm_storeu_ps(dst + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(hi, zero)), scale));
    _mm_storeu_ps(dst + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, zero)), scale));
}
#endif

void convertRowRgb(const uint8_t* src, const PixelLayout& layout, int width, float* r, float* g, float* b)
{
    constexpr float normalize = 1.0f / 255.0f;
    int x = 0;
#ifdef YOLOORTCML_SSSE3
    if (layout.step == 3) {
        // deinterleave 16 pixels of 3-byte data into per-byte streams, then scale to [0, 1]
        const __m128i m00 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i m01 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
        const __m128i m02 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
        const __m128i m10 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i m11 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
        const __m128i m12 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
        const __m128i m20 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i m21 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
        const __m128i m22 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
        const __m128 scale = _mm_set1_ps(normalize);
        float* bytePlane[3];
        bytePlane[layout.r] = r;
        bytePlane[layout.g] = g;
        bytePlane[layout.b] = b;
        for (; x + 16 <= width; x += 16) {
            const uint8_t* p = src + static_cast<size_t>(x) * 3;
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
            const __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));
            storeU8AsFloat(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, m00), _mm_shuffle_epi8(v1, m01)),
                                        _mm_shuffle_epi8(v2, m02)),
                           scale, bytePlane[0] + x);
            storeU8AsFloat(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, m10), _mm_shuffle_epi8(v1, m11)),
                                        _mm_shuffle_epi8(v2, m12)),
                           scale, bytePlane[1] + x);
            storeU8AsFloat(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v0, m20), _mm_shuffle_epi8(v1, m21)),
                                        _mm_shuffle_epi8(v2, m22)),
                           scale, bytePlane[2] + x);
        }
    }
#endif
    for (; x < width; ++x) {
        const uint8_t* p = src + static_cast<size_t>(x) * layout.step;
        r[x] = p[layout.r] * normalize;
        g[x] = p[layout.g] * normalize;
        b[x] = p[layout.b] * normalize;
    }
}

void convertRowLuma(const uint8_t* src, const PixelLayout& layout, int width, float* dst)
{
    constexpr float normalize = 1.0f / 255.0f;
    for (int x = 0; x < width; ++x, src += layout.step) {
        dst[x] = (0.299f * src[layout.r] + 0.587f * src[layout.g] + 0.114f * src[layout.b]) * normalize;
    }
}

// rows for baked-preprocessing models: interleaved RGB (or gray) uint8
void convertRowRgbU8(const uint8_t* src, const PixelLayout& layout, int width, uint8_t* dst)
{
    if (layout.step == 3 && layout.r == 0) {
        std::memcpy(dst, src, static_cast<size_t>(width) * 3);
        return;
    }
    int x = 0;
#ifdef YOLOORTCML_SSSE3
    if (layout.step == 3) {
        // BGR -> RGB, 5 pixels per iteration; lane 15 is rewritten by the next store or the tail
        const __m128i swap = _mm_setr_epi8(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
        for (; x + 6 <= width; x += 5) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + static_cast<size_t>(x) * 3));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + static_cast<size_t>(x) * 3), _mm_shuffle_epi8(v, swap));
        }
    }
#endif
    for (; x < width; ++x) {
        const uint8_t* p = src + static_cast<size_t>(x) * layout.step;
        uint8_t* q = dst + static_cast<size_t>(x) * 3;
        q[0] = p[layout.r];
        q[1] = p[layout.g];
        q[2] = p[layout.b];
    }
}

void convertRowLumaU8(const uint8_t* src, const PixelLayout& layout, int width, uint8_t* dst)
{
    if (layout.step == 1) {
        std::memcpy(dst, src, static_cast<size_t>(width));
        return;
    }
    for (int x = 0; x < width; ++x, src += layout.step) {
        dst[x] = static_cast<uint8_t>(0.299f * src[layout.r] + 0.587f * src[layout.g] + 0.114f * src[layout.b] + 0.5f);
    }
}

// interleaved RGB float row for baked-preprocessing models; a runtime pshufb mask
// reorders source bytes so BGR/RGB/BGRA/RGBA share one SIMD path
void hresizeRgbInterleaved(const uint8_t* src, const PixelLayout& layout, const PreprocessContext& context,
                           int width, float* dst)
{
    const int* offset0 = context.xOffset0.data();
    const int* offset1 = context.xOffset1.data();
    const float* weights = context.xWeight.data();
    int x = 0;
#ifdef YOLOORTCML_SSSE3
    if (layout.step == 3 || layout.step == 4) {
        const int simdEnd = layout.step == 3 ? context.xSafe : width;
        const __m128i zero = _mm_setzero_si128();
        const __m128i reorder = _mm_setr_epi8(
            static_cast<char>(layout.r), static_cast<char>(layout.g), static_cast<char>(layout.b), -1,
            static_cast<char>(8 + layout.r), static_cast<char>(8 + layout.g), static_cast<char>(8 + layout.b), -1,
            -1, -1, -1, -1, -1, -1, -1, -1);
        for (; x < simdEnd; ++x) {
            int32_t pack0;
            int32_t pack1;
            std::memcpy(&pack0, src + offset0[x], sizeof(pack0));
            std::memcpy(&pack1, src + offset1[x], sizeof(pack1));
            const __m128i pair = _mm_unpacklo_epi64(_mm_cvtsi32_si128(pack0), _mm_cvtsi32_si128(pack1));
            const __m128i rgb16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(pair, reorder), zero);
            const __m128 p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgb16, zero));
            const __m128 p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgb16, zero));
            const __m128 value = _mm_add_ps(p0, _mm_mul_ps(_mm_sub_ps(p1, p0), _mm_set1_ps(weights[x])));
            _mm_storeu_ps(dst + static_cast<size_t>(x) * 3, value);    // 4th lane is scratch, rows have one float of slack
        }
    }
#endif
    for (; x < width; ++x) {
        const uint8_t* p0 = src + offset0[x];
        const uint8_t* p1 = src + offset1[x];
        const float w = weights[x];
        float* q = dst + static_cast<size_t>(x) * 3;
        q[0] = static_cast<float>(p0[layout.r]) + (static_cast<float>(p1[layout.r]) - static_cast<float>(p0[layout.r])) * w;
        q[1] = static_cast<float>(p0[layout.g]) + (static_cast<float>(p1[layout.g]) - static_cast<float>(p0[layout.g])) * w;
        q[2] = static_cast<float>(p0[layout.b]) + (static_cast<float>(p1[layout.b]) - static_cast<float>(p0[layout.b])) * w;
    }
}

void lerpRowToU8(const float* row0, const float* row1, float weight, int count, uint8_t* dst)
{
    int i = 0;
#ifdef YOLOORTCML_SSE
    const __m128 w = _mm_set1_ps(weight);
    const __m128 half = _mm_set1_ps(0.5f);
    for (; i + 16 <= count; i += 16) {
        __m128i quads[4];
        for (int k = 0; k < 4; ++k) {
            const __m128 a = _mm_loadu_ps(row0 + i + k * 4);
            const __m128 b = _mm_loadu_ps(row1 + i + k * 4);
            quads[k] = _mm_cvttps_epi32(_mm_add_ps(_mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), w)), half));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_packus_epi16(_mm_packs_epi32(quads[0], quads[1]), _mm_packs_epi32(quads[2], quads[3])));
    }
#endif
    for (; i < count; ++i) {
        dst[i] = static_cast<uint8_t>(row0[i] + (row1[i] - row0[i]) * weight + 0.5f);
    }
}

void floatToHalf(const float* src, uint16_t* dst, size_t count)
{
    size_t i = 0;
#ifdef YOLOORTCML_F16C
    for (; i + 8 <= count; i += 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    }
#endif
    for (; i < count; ++i) {
        dst[i] = floatToHalfBits(src[i]);
    }
}
}

PreprocessResult preprocess(
    const ModelInputInfo& info,
    const ImageView& image,
    PreprocessContext& context,
    InputTensor& tensor)
{
    PreprocessResult result;
    result.imageWidth = image.width;
    result.imageHeight = image.height;

    const PixelLayout layout = pixelLayout(image.format);
    const uint8_t* pixels = static_cast<const uint8_t*>(image.data);
    const size_t stride = image.stride != 0 ? image.stride : static_cast<size_t>(image.width) * layout.step;

    const float scale = std::min(
        static_cast<float>(info.width) / static_cast<float>(image.width),
        static_cast<float>(info.height) / static_cast<float>(image.height));
    const int resizedWidth = std::max(1, static_cast<int>(std::lround(image.width * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::lround(image.height * scale)));
    int tensorWidth = info.width;
    int tensorHeight = info.height;
    if (info.dynamicSize) {
        tensorWidth = std::max(32, alignUp(resizedWidth, 32));
        tensorHeight = std::max(32, alignUp(resizedHeight, 32));
    }

    result.invScale = 1.0f / scale;
    result.padX = static_cast<float>(tensorWidth - resizedWidth) * 0.5f;
    result.padY = static_cast<float>(tensorHeight - resizedHeight) * 0.5f;
    const int padLeft = static_cast<int>(std::lround(result.padX - 0.1f));
    const int padTop = static_cast<int>(std::lround(result.padY - 0.1f));

    tensor.raw = info.rawInput;
    tensor.fp16 = !info.rawInput && info.fp16;
    const size_t area = static_cast<size_t>(tensorWidth) * static_cast<size_t>(tensorHeight);
    const size_t count = area * info.channels;
    const bool padded = resizedWidth != tensorWidth || resizedHeight != tensorHeight;
    uint8_t* bytePixels = nullptr;
    float* planes = nullptr;
    uint16_t* halfPlanes = nullptr;
    if (info.rawInput) {
        tensor.shape = {1, tensorHeight, tensorWidth, info.channels};
        tensor.byteData.resize(count);
        bytePixels = tensor.byteData.data();
        if (padded) {
            std::fill(tensor.byteData.begin(), tensor.byteData.end(), static_cast<uint8_t>(114));
        }
    } else {
        tensor.shape = {1, info.channels, tensorHeight, tensorWidth};
        tensor.floatData.resize(count);    // fp16 models stage each row here before conversion
        planes = tensor.floatData.data();
        if (info.fp16) {
            tensor.halfData.resize(count);
            halfPlanes = tensor.halfData.data();
        }
        if (padded) {
            if (info.fp16) {
                std::fill(tensor.halfData.begin(), tensor.halfData.end(), floatToHalfBits(114.0f / 255.0f));
            } else {
                std::fill(tensor.floatData.begin(), tensor.floatData.end(), 114.0f / 255.0f);
            }
        }
    }
    if (resizedWidth == image.width && resizedHeight == image.height) {
        for (int y = 0; y < resizedHeight; ++y) {
            const uint8_t* src = pixels + static_cast<size_t>(y) * stride;
            if (info.rawInput) {
                uint8_t* dst = bytePixels + (static_cast<size_t>(y + padTop) * tensorWidth + padLeft) * info.channels;
                if (info.channels == 3) {
                    convertRowRgbU8(src, layout, image.width, dst);
                } else {
                    convertRowLumaU8(src, layout, image.width, dst);
                }
                continue;
            }
            const size_t offset = static_cast<size_t>(y + padTop) * tensorWidth + padLeft;
            float* dst = planes + offset;
            if (info.channels == 3) {
                convertRowRgb(src, layout, image.width, dst, dst + area, dst + 2 * area);
            } else {
                convertRowLuma(src, layout, image.width, dst);
            }
            if (halfPlanes != nullptr) {
                for (int c = 0; c < info.channels; ++c) {
                    floatToHalf(dst + static_cast<size_t>(c) * area,
                                halfPlanes + offset + static_cast<size_t>(c) * area,
                                image.width);
                }
            }
        }
    } else {
        buildXMap(image.width, resizedWidth, layout.step, context);
        context.rowA.resize(static_cast<size_t>(resizedWidth) * info.channels + 1);    // one float of slack for SIMD stores
        context.rowB.resize(static_cast<size_t>(resizedWidth) * info.channels + 1);
        float* bufferA = context.rowA.data();
        float* bufferB = context.rowB.data();
        int rowInA = -1;
        int rowInB = -1;

        const auto horizontalResize = [&](int srcRow, float* dst) {
            const uint8_t* src = pixels + static_cast<size_t>(srcRow) * stride;
            if (info.rawInput && info.channels == 3) {
                hresizeRgbInterleaved(src, layout, context, resizedWidth, dst);
            } else if (info.channels == 3) {
                hresizeRgb(src, layout, context, resizedWidth, dst, dst + resizedWidth, dst + 2 * resizedWidth);
            } else {
                hresizeLuma(src, layout, context, resizedWidth, dst);
            }
        };
        const auto resolveRow = [&](int srcRow, int keepRow) -> const float* {
            if (srcRow == rowInA) {
                return bufferA;
            }
            if (srcRow == rowInB) {
                return bufferB;
            }
            float* target = rowInA == keepRow ? bufferB : bufferA;
            horizontalResize(srcRow, target);
            (target == bufferA ? rowInA : rowInB) = srcRow;
            return target;
        };

        const float ratio = static_cast<float>(image.height) / static_cast<float>(resizedHeight);
        for (int y = 0; y < resizedHeight; ++y) {
            const float srcY = (static_cast<float>(y) + 0.5f) * ratio - 0.5f;
            int y0 = static_cast<int>(std::floor(srcY));
            float weight = srcY - static_cast<float>(y0);
            if (y0 < 0) {
                y0 = 0;
                weight = 0.0f;
            }
            const int y1 = std::min(y0 + 1, image.height - 1);

            const float* row0 = resolveRow(y0, y1);
            const float* row1 = resolveRow(y1, y0);
            if (info.rawInput) {
                uint8_t* dst = bytePixels + (static_cast<size_t>(y + padTop) * tensorWidth + padLeft) * info.channels;
                lerpRowToU8(row0, row1, weight, resizedWidth * info.channels, dst);
                continue;
            }
            const size_t offset = static_cast<size_t>(y + padTop) * tensorWidth + padLeft;
            float* dst = planes + offset;
            for (int c = 0; c < info.channels; ++c) {
                vresizeStore(
                    row0 + static_cast<size_t>(c) * resizedWidth,
                    row1 + static_cast<size_t>(c) * resizedWidth,
                    weight,
                    resizedWidth,
                    dst + static_cast<size_t>(c) * area);
            }
            if (halfPlanes != nullptr) {
                for (int c = 0; c < info.channels; ++c) {
                    floatToHalf(dst + static_cast<size_t>(c) * area,
                                halfPlanes + offset + static_cast<size_t>(c) * area,
                                resizedWidth);
                }
            }
        }
    }

    return result;
}
