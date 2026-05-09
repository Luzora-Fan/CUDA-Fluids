#include "Fluid3D.cuh"
#include "Multigrid.cuh"

/**
 * @file Fluid3D.cu
 * @brief CUDA kernels and host-side GPU resource management for Fluid3D.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlock = 8;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA call failed: ") + expression + " at " + file + ":" + std::to_string(line) +
            " (" + cudaGetErrorString(result) + ")");
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

dim3 grid3(int w, int h, int d) {
    return dim3((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock, (d + kBlock - 1) / kBlock);
}

__device__ inline int idx3(int x, int y, int z, int w, int h) {
    return (z * h + y) * w + x;
}

__device__ inline int clampInt(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float clampF(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__device__ float sampleScalar3(const float* f, int w, int h, int d, float x, float y, float z) {
    x = clampF(x, 0.0f, (float)(w - 1));
    y = clampF(y, 0.0f, (float)(h - 1));
    z = clampF(z, 0.0f, (float)(d - 1));
    const int x0 = (int)floorf(x), y0 = (int)floorf(y), z0 = (int)floorf(z);
    const int x1 = clampInt(x0 + 1, 0, w - 1);
    const int y1 = clampInt(y0 + 1, 0, h - 1);
    const int z1 = clampInt(z0 + 1, 0, d - 1);
    const float tx = x - (float)x0, ty = y - (float)y0, tz = z - (float)z0;

    const float c000 = f[idx3(x0, y0, z0, w, h)];
    const float c100 = f[idx3(x1, y0, z0, w, h)];
    const float c010 = f[idx3(x0, y1, z0, w, h)];
    const float c110 = f[idx3(x1, y1, z0, w, h)];
    const float c001 = f[idx3(x0, y0, z1, w, h)];
    const float c101 = f[idx3(x1, y0, z1, w, h)];
    const float c011 = f[idx3(x0, y1, z1, w, h)];
    const float c111 = f[idx3(x1, y1, z1, w, h)];
    const float c00 = c000 + tx * (c100 - c000);
    const float c10 = c010 + tx * (c110 - c010);
    const float c01 = c001 + tx * (c101 - c001);
    const float c11 = c011 + tx * (c111 - c011);
    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);
    return c0 + tz * (c1 - c0);
}

__device__ float3 sampleVec3(const float3* f, int w, int h, int d, float x, float y, float z) {
    x = clampF(x, 0.0f, (float)(w - 1));
    y = clampF(y, 0.0f, (float)(h - 1));
    z = clampF(z, 0.0f, (float)(d - 1));
    const int x0 = (int)floorf(x), y0 = (int)floorf(y), z0 = (int)floorf(z);
    const int x1 = clampInt(x0 + 1, 0, w - 1);
    const int y1 = clampInt(y0 + 1, 0, h - 1);
    const int z1 = clampInt(z0 + 1, 0, d - 1);
    const float tx = x - (float)x0, ty = y - (float)y0, tz = z - (float)z0;

    const float3 c000 = f[idx3(x0, y0, z0, w, h)];
    const float3 c100 = f[idx3(x1, y0, z0, w, h)];
    const float3 c010 = f[idx3(x0, y1, z0, w, h)];
    const float3 c110 = f[idx3(x1, y1, z0, w, h)];
    const float3 c001 = f[idx3(x0, y0, z1, w, h)];
    const float3 c101 = f[idx3(x1, y0, z1, w, h)];
    const float3 c011 = f[idx3(x0, y1, z1, w, h)];
    const float3 c111 = f[idx3(x1, y1, z1, w, h)];

    #define L3(A, B, T) make_float3((A).x + (T) * ((B).x - (A).x), (A).y + (T) * ((B).y - (A).y), (A).z + (T) * ((B).z - (A).z))
    const float3 c00 = L3(c000, c100, tx);
    const float3 c10 = L3(c010, c110, tx);
    const float3 c01 = L3(c001, c101, tx);
    const float3 c11 = L3(c011, c111, tx);
    const float3 c0 = L3(c00, c10, ty);
    const float3 c1 = L3(c01, c11, ty);
    return L3(c0, c1, tz);
    #undef L3
}

__global__ void addImpulseKernel(
    float* density, float* temperature, float3* velocity,
    int w, int h, int d,
    float nx, float ny, float nz, float nr,
    float densityAmount, float temperatureAmount, float3 velocityAmount) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;

    const float cx = nx * (float)(w - 1);
    const float cy = ny * (float)(h - 1);
    const float cz = nz * (float)(d - 1);
    const int shortest = min(min(w, h), d);
    const float radius = fmaxf(1.0f, nr * (float)shortest);
    const float dx = (float)x - cx, dy = (float)y - cy, dz = (float)z - cz;
    const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist > radius) return;
    const float t = 1.0f - dist / radius;
    const float falloff = t * t * (3.0f - 2.0f * t);
    const int id = idx3(x, y, z, w, h);
    density[id] += densityAmount * falloff;
    temperature[id] += temperatureAmount * falloff;
    velocity[id].x += velocityAmount.x * falloff;
    velocity[id].y += velocityAmount.y * falloff;
    velocity[id].z += velocityAmount.z * falloff;
}

__global__ void advectVelocityKernel(
    const float3* velField, const float3* src, float3* dst,
    int w, int h, int d, float dt, float dissipation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    const float3 v = velField[id];
    const float px = (float)x - dt * v.x;
    const float py = (float)y - dt * v.y;
    const float pz = (float)z - dt * v.z;
    const float3 s = sampleVec3(src, w, h, d, px, py, pz);
    dst[id] = make_float3(s.x * dissipation, s.y * dissipation, s.z * dissipation);
}

__global__ void advectScalarKernel(
    const float3* velField, const float* src, float* dst,
    int w, int h, int d, float dt, float dissipation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    const float3 v = velField[id];
    const float px = (float)x - dt * v.x;
    const float py = (float)y - dt * v.y;
    const float pz = (float)z - dt * v.z;
    dst[id] = sampleScalar3(src, w, h, d, px, py, pz) * dissipation;
}

__global__ void applyBuoyancyKernel(float3* velocity, const float* temperature, int w, int h, int d, float dt, float strength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) return;
    const int id = idx3(x, y, z, w, h);
    velocity[id].y += dt * strength * temperature[id];
}

__global__ void enforceVelocityBoundaryKernel(float3* velocity, int w, int h, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        velocity[idx3(x, y, z, w, h)] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void computeVorticityKernel(const float3* velocity, float3* vorticity, int w, int h, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        vorticity[id] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    const float3 xl = velocity[idx3(x - 1, y, z, w, h)];
    const float3 xr = velocity[idx3(x + 1, y, z, w, h)];
    const float3 yl = velocity[idx3(x, y - 1, z, w, h)];
    const float3 yr = velocity[idx3(x, y + 1, z, w, h)];
    const float3 zl = velocity[idx3(x, y, z - 1, w, h)];
    const float3 zr = velocity[idx3(x, y, z + 1, w, h)];
    // curl: (dW/dy - dV/dz, dU/dz - dW/dx, dV/dx - dU/dy)
    const float wx = 0.5f * ((yr.z - yl.z) - (zr.y - zl.y));
    const float wy = 0.5f * ((zr.x - zl.x) - (xr.z - xl.z));
    const float wz = 0.5f * ((xr.y - xl.y) - (yr.x - yl.x));
    vorticity[id] = make_float3(wx, wy, wz);
}

__global__ void applyVorticityConfinementKernel(
    float3* velocity, const float3* vorticity, int w, int h, int d, float dt, float strength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) return;

    #define MAG(V) sqrtf((V).x * (V).x + (V).y * (V).y + (V).z * (V).z)
    const float ml = MAG(vorticity[idx3(x - 1, y, z, w, h)]);
    const float mr = MAG(vorticity[idx3(x + 1, y, z, w, h)]);
    const float md = MAG(vorticity[idx3(x, y - 1, z, w, h)]);
    const float mu = MAG(vorticity[idx3(x, y + 1, z, w, h)]);
    const float mb = MAG(vorticity[idx3(x, y, z - 1, w, h)]);
    const float mf = MAG(vorticity[idx3(x, y, z + 1, w, h)]);
    #undef MAG

    float gx = 0.5f * (mr - ml);
    float gy = 0.5f * (mu - md);
    float gz = 0.5f * (mf - mb);
    const float glen = sqrtf(gx * gx + gy * gy + gz * gz) + 1.0e-5f;
    gx /= glen; gy /= glen; gz /= glen;

    const int id = idx3(x, y, z, w, h);
    const float3 omega = vorticity[id];
    // force = N x omega
    const float fx = gy * omega.z - gz * omega.y;
    const float fy = gz * omega.x - gx * omega.z;
    const float fz = gx * omega.y - gy * omega.x;
    velocity[id].x += dt * strength * fx;
    velocity[id].y += dt * strength * fy;
    velocity[id].z += dt * strength * fz;
}

/**
 * One Jacobi sweep for implicit viscous diffusion: solve (I - alpha*L) u = u0
 * via u_new = (u0 + alpha * sum_neighbors(u)) / (1 + 6 * alpha) where
 * alpha = viscosity * dt (unit grid spacing).
 */
__global__ void diffuseVelocityJacobiKernel(
    const float3* u0, const float3* uIn, float3* uOut, int w, int h, int d, float alpha) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        uOut[id] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    const float3 l = uIn[idx3(x - 1, y, z, w, h)];
    const float3 r = uIn[idx3(x + 1, y, z, w, h)];
    const float3 dn = uIn[idx3(x, y - 1, z, w, h)];
    const float3 up = uIn[idx3(x, y + 1, z, w, h)];
    const float3 bk = uIn[idx3(x, y, z - 1, w, h)];
    const float3 fr = uIn[idx3(x, y, z + 1, w, h)];
    const float3 b = u0[id];
    const float denom = 1.0f + 6.0f * alpha;
    uOut[id].x = (b.x + alpha * (l.x + r.x + dn.x + up.x + bk.x + fr.x)) / denom;
    uOut[id].y = (b.y + alpha * (l.y + r.y + dn.y + up.y + bk.y + fr.y)) / denom;
    uOut[id].z = (b.z + alpha * (l.z + r.z + dn.z + up.z + bk.z + fr.z)) / denom;
}

__global__ void computeDivergenceKernel(const float3* velocity, float* divergence, int w, int h, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        divergence[id] = 0.0f;
        return;
    }
    const float3 xl = velocity[idx3(x - 1, y, z, w, h)];
    const float3 xr = velocity[idx3(x + 1, y, z, w, h)];
    const float3 yl = velocity[idx3(x, y - 1, z, w, h)];
    const float3 yr = velocity[idx3(x, y + 1, z, w, h)];
    const float3 zl = velocity[idx3(x, y, z - 1, w, h)];
    const float3 zr = velocity[idx3(x, y, z + 1, w, h)];
    divergence[id] = 0.5f * ((xr.x - xl.x) + (yr.y - yl.y) + (zr.z - zl.z));
}

__global__ void projectVelocityKernel(float3* velocity, const float* pressure, int w, int h, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        velocity[id] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    const float xl = pressure[idx3(x - 1, y, z, w, h)];
    const float xr = pressure[idx3(x + 1, y, z, w, h)];
    const float yl = pressure[idx3(x, y - 1, z, w, h)];
    const float yr = pressure[idx3(x, y + 1, z, w, h)];
    const float zl = pressure[idx3(x, y, z - 1, w, h)];
    const float zr = pressure[idx3(x, y, z + 1, w, h)];
    velocity[id].x -= 0.5f * (xr - xl);
    velocity[id].y -= 0.5f * (yr - yl);
    velocity[id].z -= 0.5f * (zr - zl);
}

}  // namespace

Fluid3D::Fluid3D(FluidSettings settings) : settings_(settings) {
    if (settings_.width < 4 || settings_.height < 4 || settings_.depth < 4) {
        throw std::invalid_argument("Fluid grid must be at least 4x4x4.");
    }
    if (settings_.dt <= 0.0f) {
        throw std::invalid_argument("Timestep must be positive.");
    }
    if (settings_.mgLevels < 1) {
        throw std::invalid_argument("mgLevels must be at least 1.");
    }
    const int factor = 1 << (settings_.mgLevels - 1);
    if ((settings_.width % factor) || (settings_.height % factor) || (settings_.depth % factor)) {
        throw std::invalid_argument(
            "Grid dimensions must each be divisible by 2^(mgLevels-1) = " + std::to_string(factor));
    }
    if (settings_.viscosity < 0.0f || settings_.viscosityIterations < 0) {
        throw std::invalid_argument("Viscosity parameters must be non-negative.");
    }

    CUDA_CHECK(cudaMalloc(&velocity_, vectorBytes()));
    CUDA_CHECK(cudaMalloc(&velocityScratch_, vectorBytes()));
    CUDA_CHECK(cudaMalloc(&density_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&densityScratch_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&temperature_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&temperatureScratch_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&pressure_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&divergence_, scalarBytes()));
    CUDA_CHECK(cudaMalloc(&vorticity_, vectorBytes()));

    // Build multigrid hierarchy. Level 0 aliases the main pressure/divergence buffers.
    mgPressure_.push_back(pressure_);
    mgRhs_.push_back(divergence_);
    mgWidths_.push_back(settings_.width);
    mgHeights_.push_back(settings_.height);
    mgDepths_.push_back(settings_.depth);
    for (int L = 1; L < settings_.mgLevels; ++L) {
        const int w = mgWidths_.back() / 2;
        const int h = mgHeights_.back() / 2;
        const int d = mgDepths_.back() / 2;
        const size_t bytes = sizeof(float) * (size_t)w * h * d;
        float* p = nullptr;
        float* r = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        CUDA_CHECK(cudaMalloc(&r, bytes));
        mgPressure_.push_back(p);
        mgRhs_.push_back(r);
        mgWidths_.push_back(w);
        mgHeights_.push_back(h);
        mgDepths_.push_back(d);
    }

    reset();
}

Fluid3D::~Fluid3D() {
    cudaFree(velocity_);
    cudaFree(velocityScratch_);
    cudaFree(density_);
    cudaFree(densityScratch_);
    cudaFree(temperature_);
    cudaFree(temperatureScratch_);
    cudaFree(pressure_);
    cudaFree(divergence_);
    cudaFree(vorticity_);
    for (size_t i = 1; i < mgPressure_.size(); ++i) {
        cudaFree(mgPressure_[i]);
        cudaFree(mgRhs_[i]);
    }
}

void Fluid3D::reset() {
    CUDA_CHECK(cudaMemset(velocity_, 0, vectorBytes()));
    CUDA_CHECK(cudaMemset(velocityScratch_, 0, vectorBytes()));
    CUDA_CHECK(cudaMemset(density_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(densityScratch_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(temperature_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(temperatureScratch_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(pressure_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(divergence_, 0, scalarBytes()));
    CUDA_CHECK(cudaMemset(vorticity_, 0, vectorBytes()));
}

void Fluid3D::addImpulse(
    float nx, float ny, float nz, float nr,
    float densityAmount, float temperatureAmount, float3 velocityAmount) {
    const dim3 block(kBlock, kBlock, kBlock);
    addImpulseKernel<<<grid3(settings_.width, settings_.height, settings_.depth), block>>>(
        density_, temperature_, velocity_,
        settings_.width, settings_.height, settings_.depth,
        std::clamp(nx, 0.0f, 1.0f),
        std::clamp(ny, 0.0f, 1.0f),
        std::clamp(nz, 0.0f, 1.0f),
        std::max(nr, 0.0f),
        densityAmount, temperatureAmount, velocityAmount);
    CUDA_CHECK(cudaGetLastError());
}

void Fluid3D::step() {
    const dim3 block(kBlock, kBlock, kBlock);
    const dim3 grid = grid3(settings_.width, settings_.height, settings_.depth);
    const int W = settings_.width, H = settings_.height, D = settings_.depth;

    // 1. Advect velocity through itself.
    advectVelocityKernel<<<grid, block>>>(
        velocity_, velocity_, velocityScratch_, W, H, D, settings_.dt, settings_.velocityDissipation);
    std::swap(velocity_, velocityScratch_);

    // 2. Buoyancy from temperature.
    if (settings_.buoyancyStrength > 0.0f) {
        applyBuoyancyKernel<<<grid, block>>>(velocity_, temperature_, W, H, D, settings_.dt, settings_.buoyancyStrength);
    }

    enforceVelocityBoundaryKernel<<<grid, block>>>(velocity_, W, H, D);

    // 3. Vortex confinement.
    if (settings_.vorticityStrength > 0.0f) {
        computeVorticityKernel<<<grid, block>>>(velocity_, vorticity_, W, H, D);
        applyVorticityConfinementKernel<<<grid, block>>>(
            velocity_, vorticity_, W, H, D, settings_.dt, settings_.vorticityStrength);
        enforceVelocityBoundaryKernel<<<grid, block>>>(velocity_, W, H, D);
    }

    // 4. Implicit viscous diffusion via Jacobi.
    if (settings_.viscosity > 0.0f && settings_.viscosityIterations > 0) {
        const float alpha = settings_.viscosity * settings_.dt;
        // Copy current velocity into scratch as the initial guess and as u0.
        CUDA_CHECK(cudaMemcpy(velocityScratch_, velocity_, vectorBytes(), cudaMemcpyDeviceToDevice));
        // Allocate a second iterate buffer; reuse vorticity_ would work in size (vector field) but it
        // holds curl. Use a small temporary.
        float3* uTmp = nullptr;
        CUDA_CHECK(cudaMalloc(&uTmp, vectorBytes()));
        CUDA_CHECK(cudaMemcpy(uTmp, velocity_, vectorBytes(), cudaMemcpyDeviceToDevice));
        // After iterations, the solution should land in `velocity_`.
        // Iterate: read from `uTmp` (or velocity_), write to the other. velocityScratch_ holds u0.
        float3* uIn = velocity_;
        float3* uOut = uTmp;
        for (int i = 0; i < settings_.viscosityIterations; ++i) {
            diffuseVelocityJacobiKernel<<<grid, block>>>(velocityScratch_, uIn, uOut, W, H, D, alpha);
            std::swap(uIn, uOut);
        }
        // uIn holds the final result. Ensure it ends up in velocity_.
        if (uIn != velocity_) {
            CUDA_CHECK(cudaMemcpy(velocity_, uIn, vectorBytes(), cudaMemcpyDeviceToDevice));
        }
        cudaFree(uTmp);
        enforceVelocityBoundaryKernel<<<grid, block>>>(velocity_, W, H, D);
    }

    // 5. Divergence and multigrid pressure solve.
    computeDivergenceKernel<<<grid, block>>>(velocity_, divergence_, W, H, D);
    CUDA_CHECK(cudaMemset(pressure_, 0, scalarBytes()));
    mg::runVCycles(
        mgPressure_, mgRhs_, mgWidths_, mgHeights_, mgDepths_,
        settings_.mgPreSmooth, settings_.mgPostSmooth, settings_.mgCoarseIterations, settings_.mgVCycles);

    // 6. Project velocity.
    projectVelocityKernel<<<grid, block>>>(velocity_, pressure_, W, H, D);
    enforceVelocityBoundaryKernel<<<grid, block>>>(velocity_, W, H, D);

    // 7. Advect density and temperature.
    advectScalarKernel<<<grid, block>>>(
        velocity_, density_, densityScratch_, W, H, D, settings_.dt, settings_.densityDissipation);
    std::swap(density_, densityScratch_);
    advectScalarKernel<<<grid, block>>>(
        velocity_, temperature_, temperatureScratch_, W, H, D, settings_.dt, settings_.temperatureDissipation);
    std::swap(temperature_, temperatureScratch_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<float> Fluid3D::copyDensityToHost() const {
    std::vector<float> h(static_cast<size_t>(cellCount()));
    CUDA_CHECK(cudaMemcpy(h.data(), density_, scalarBytes(), cudaMemcpyDeviceToHost));
    return h;
}

std::vector<float> Fluid3D::copyTemperatureToHost() const {
    std::vector<float> h(static_cast<size_t>(cellCount()));
    CUDA_CHECK(cudaMemcpy(h.data(), temperature_, scalarBytes(), cudaMemcpyDeviceToHost));
    return h;
}

int Fluid3D::cellCount() const { return settings_.width * settings_.height * settings_.depth; }
size_t Fluid3D::scalarBytes() const { return (size_t)cellCount() * sizeof(float); }
size_t Fluid3D::vectorBytes() const { return (size_t)cellCount() * sizeof(float3); }
