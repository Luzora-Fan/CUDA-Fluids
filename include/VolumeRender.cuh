#pragma once

/**
 * @file VolumeRender.cuh
 * @brief Maximum-intensity-projection volume renderer for the 3D solver.
 *
 * Projects the 3D density and temperature volumes along the z axis onto a
 * 2D RGBA8 image, applying exposure to map smoke to red and heat to green.
 * Output is a small (width*height*4) device buffer that the caller copies
 * to the host and uploads as an OpenGL texture.
 */

#include <cuda_runtime.h>

namespace volr {

/**
 * @brief Owns a device RGBA8 buffer sized width * height * 4.
 */
class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /** Renders MIP along z; output is width*height RGBA8 on device. */
    void render(const float* density, const float* temperature, int w, int h, int d,
                float densityExposure, float temperatureExposure);

    /** Copies the rendered RGBA8 image into a host buffer (size width*height*4). */
    void copyToHost(unsigned char* dst) const;

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;
    unsigned char* image_ = nullptr;
};

}  // namespace volr
