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

__global__ void mipKernel(
    const float* density, const float* temperature,
    unsigned char* image, int w, int h, int d,
    float densityExposure, float temperatureExposure) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float maxDensity = 0.0f;
    float maxTemperature = 0.0f;
    for (int z = 0; z < d; ++z) {
        const int id = idx3(x, y, z, w, h);
        maxDensity = fmaxf(maxDensity, density[id]);
        maxTemperature = fmaxf(maxTemperature, temperature[id]);
    }

    const float smoke = clampF(1.0f - expf(-fmaxf(maxDensity, 0.0f) * densityExposure), 0.0f, 1.0f);
    const float heat = clampF(1.0f - expf(-fmaxf(maxTemperature, 0.0f) * temperatureExposure), 0.0f, 1.0f);

    const int pixel = (y * w + x) * 4;
    image[pixel + 0] = (unsigned char)(smoke * 255.0f);
    image[pixel + 1] = (unsigned char)(heat * 255.0f);
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
                     float densityExposure, float temperatureExposure) {
    if (w != width_ || h != height_) {
        throw std::runtime_error("VolumeRender: grid xy must match renderer size");
    }
    const dim3 block(kBlock, kBlock);
    const dim3 grid((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock);
    mipKernel<<<grid, block>>>(density, temperature, image_, w, h, d, densityExposure, temperatureExposure);
    VR_CHECK(cudaGetLastError());
}

void Renderer::copyToHost(unsigned char* dst) const {
    VR_CHECK(cudaMemcpy(dst, image_, (size_t)width_ * height_ * 4, cudaMemcpyDeviceToHost));
}

}  // namespace volr
