#include "VolumeRender.cuh"

#include <stdexcept>
#include <string>

namespace volr {

namespace {

constexpr int kBlock = 16;

void check(cudaError_t r, const char* e) {
    if (r != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + e + " (" + cudaGetErrorString(r) + ")");
    }
}

#define VR_CHECK(expr) check((expr), #expr)

__device__ inline int idx3(int x, int y, int z, int w, int h) {
    return (z * h + y) * w + x;
}

__device__ inline float clampF(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

// Cheap per-pixel hash -> [0,1). Used to jitter ray start and to dither output.
__device__ inline float hash01(unsigned int x, unsigned int y) {
    unsigned int h = x * 1973u + y * 9277u;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h = h ^ (h >> 16);
    return (float)(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Triangular PDF dither in [-1,1], from two uniform samples.
__device__ inline float triDither(float u0, float u1) {
    return u0 - u1;
}

__device__ inline unsigned char quantize(float v, float dither) {
    // dither is TPDF in [-1,1]; full 1-LSB amplitude is what fully whitens
    // 8-bit quantization noise. Anything less leaves residual banding on
    // shallow gradients (e.g. the sky).
    float q = v * 255.0f + dither + 0.5f;
    return (unsigned char)clampF(q, 0.0f, 255.0f);
}

__device__ inline float sampleTrilinear(const float* field, float fx, float fy, float fz,
                                       int w, int h, int d) {
    if (fx < 0.0f || fy < 0.0f || fz < 0.0f) return 0.0f;
    if (fx > (float)(w - 1) || fy > (float)(h - 1) || fz > (float)(d - 1)) return 0.0f;
    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const int z0 = (int)floorf(fz);
    const int x1 = min(x0 + 1, w - 1);
    const int y1 = min(y0 + 1, h - 1);
    const int z1 = min(z0 + 1, d - 1);
    const float tx = fx - (float)x0;
    const float ty = fy - (float)y0;
    const float tz = fz - (float)z0;
    const float c000 = field[idx3(x0, y0, z0, w, h)];
    const float c100 = field[idx3(x1, y0, z0, w, h)];
    const float c010 = field[idx3(x0, y1, z0, w, h)];
    const float c110 = field[idx3(x1, y1, z0, w, h)];
    const float c001 = field[idx3(x0, y0, z1, w, h)];
    const float c101 = field[idx3(x1, y0, z1, w, h)];
    const float c011 = field[idx3(x0, y1, z1, w, h)];
    const float c111 = field[idx3(x1, y1, z1, w, h)];
    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

__global__ void mipKernel(
    const float* density, const float* temperature,
    unsigned char* image, int imgW, int imgH,
    int w, int h, int d,
    float densityExposure, float temperatureExposure,
    float yaw, float pitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgW || y >= imgH) return;

    const float cy_ = cosf(yaw),   sy_ = sinf(yaw);
    const float cp_ = cosf(pitch), sp_ = sinf(pitch);
    // forward = R_y(yaw) * R_x(pitch) * (0,0,1)
    const float fwdX =  sy_ * cp_;
    const float fwdY = -sp_;
    const float fwdZ =  cy_ * cp_;
    // right = normalize(cross((0,1,0), forward))
    float rX = fwdZ;
    float rY = 0.0f;
    float rZ = -fwdX;
    float rLen = sqrtf(rX * rX + rZ * rZ);
    if (rLen < 1e-5f) { rX = 1.0f; rY = 0.0f; rZ = 0.0f; rLen = 1.0f; }
    rX /= rLen; rZ /= rLen;
    // up = cross(forward, right)
    const float uX = fwdY * rZ - fwdZ * rY;
    const float uY = fwdZ * rX - fwdX * rZ;
    const float uZ = fwdX * rY - fwdY * rX;

    const float cX = 0.5f * (float)(w - 1);
    const float cY = 0.5f * (float)(h - 1);
    const float cZ = 0.5f * (float)(d - 1);
    const float radius = 0.5f * sqrtf((float)(w * w + h * h + d * d));

    // Map image-pixel coords to grid-space units so rays span the volume
    // regardless of output resolution. When imgW==w, this matches the original
    // 1-ray-per-voxel mapping; when imgW>w, each voxel is supersampled.
    const float sx = (float)w / (float)imgW;
    const float sy = (float)h / (float)imgH;
    const float u = ((float)x - 0.5f * (float)(imgW - 1)) * sx;
    const float v = (((float)imgH - 1.0f - (float)y) - 0.5f * (float)(imgH - 1)) * sy;

    const float startX = cX - fwdX * radius + rX * u + uX * v;
    const float startY = cY - fwdY * radius + rY * u + uY * v;
    const float startZ = cZ - fwdZ * radius + rZ * u + uZ * v;

    // Step size drives MIP banding: with regular sampling, max-along-ray
    // underestimates true peaks by ~O(stepSize^2) for smooth fields and the
    // bias varies between adjacent pixels => layered bands. 0.25 voxel is a
    // good quality/perf tradeoff for 64^3-class grids with trilinear interp.
    const float stepSize = 0.25f;
    const int steps = (int)(2.0f * radius / stepSize) + 1;

    // Jitter ray start within one step to break regular-sampling banding.
    const float jitter = hash01((unsigned int)x, (unsigned int)y) * stepSize;

    float maxDensity = 0.0f;
    float maxTemperature = 0.0f;
    for (int s = 0; s < steps; ++s) {
        const float t = (float)s * stepSize + jitter;
        const float px = startX + fwdX * t;
        const float py = startY + fwdY * t;
        const float pz = startZ + fwdZ * t;
        maxDensity = fmaxf(maxDensity, sampleTrilinear(density, px, py, pz, w, h, d));
        maxTemperature = fmaxf(maxTemperature, sampleTrilinear(temperature, px, py, pz, w, h, d));
    }

    // Reinhard tone map: x*k / (1 + x*k). Gentler rolloff than 1-exp(-x*k),
    // so the bright core of the plume keeps more headroom and doesn't clip
    // a wide range of input values onto the same output codes.
    const float dk = fmaxf(maxDensity, 0.0f) * densityExposure;
    const float tk = fmaxf(maxTemperature, 0.0f) * temperatureExposure;
    const float smoke = clampF(dk / (1.0f + dk), 0.0f, 1.0f);
    const float heat  = clampF(tk / (1.0f + tk), 0.0f, 1.0f);

    // Triangular-PDF dither (decorrelated per channel) to mask 8-bit banding.
    const float d0 = hash01((unsigned int)x ^ 0x9E3779B1u, (unsigned int)y);
    const float d1 = hash01((unsigned int)x, (unsigned int)y ^ 0x85EBCA77u);
    const float d2 = hash01((unsigned int)x ^ 0xC2B2AE3Du, (unsigned int)y ^ 0x27D4EB2Fu);
    const float d3 = hash01((unsigned int)x ^ 0x165667B1u, (unsigned int)y ^ 0xD3A2646Cu);

    const int pixel = (y * imgW + x) * 4;
    image[pixel + 0] = quantize(smoke, triDither(d0, d1));
    image[pixel + 1] = quantize(heat,  triDither(d2, d3));
    image[pixel + 2] = 0;
    image[pixel + 3] = 255;
}

}  // namespace

Renderer::Renderer(int width, int height) : width_(width), height_(height) {
    VR_CHECK(cudaMalloc(&image_, (size_t)width * height * 4));
    VR_CHECK(cudaMemset(image_, 0, (size_t)width * height * 4));
}

Renderer::~Renderer() {
    cudaFree(image_);
}

void Renderer::render(const float* density, const float* temperature, int w, int h, int d,
                     float densityExposure, float temperatureExposure,
                     float yaw, float pitch) {
    const dim3 block(kBlock, kBlock);
    const dim3 grid((width_ + kBlock - 1) / kBlock, (height_ + kBlock - 1) / kBlock);
    mipKernel<<<grid, block>>>(density, temperature, image_, width_, height_, w, h, d,
                               densityExposure, temperatureExposure, yaw, pitch);
    VR_CHECK(cudaGetLastError());
}

void Renderer::copyToHost(unsigned char* dst) const {
    VR_CHECK(cudaMemcpy(dst, image_, (size_t)width_ * height_ * 4, cudaMemcpyDeviceToHost));
}

}  // namespace volr
