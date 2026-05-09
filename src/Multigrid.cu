#include "Multigrid.cuh"

/**
 * @file Multigrid.cu
 * @brief Geometric multigrid kernels for the 3D pressure Poisson equation.
 */

#include <stdexcept>
#include <string>

namespace mg {

namespace {

constexpr int kBlock = 8;

void check(cudaError_t result, const char* expr) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + expr + " (" + cudaGetErrorString(result) + ")");
    }
}

#define MG_CHECK(expr) check((expr), #expr)

__device__ inline int idx3(int x, int y, int z, int w, int h) {
    return (z * h + y) * w + x;
}

dim3 grid3(int w, int h, int d) {
    return dim3((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock, (d + kBlock - 1) / kBlock);
}

__global__ void smoothKernel(float* pressure, const float* rhs, int w, int h, int d, int parity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;
    if (((x + y + z) & 1) != parity) return;

    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        pressure[id] = 0.0f;
        return;
    }
    const float l = pressure[idx3(x - 1, y, z, w, h)];
    const float r = pressure[idx3(x + 1, y, z, w, h)];
    const float dn = pressure[idx3(x, y - 1, z, w, h)];
    const float up = pressure[idx3(x, y + 1, z, w, h)];
    const float bk = pressure[idx3(x, y, z - 1, w, h)];
    const float fr = pressure[idx3(x, y, z + 1, w, h)];
    pressure[id] = (l + r + dn + up + bk + fr - rhs[id]) * (1.0f / 6.0f);
}

__global__ void residualKernel(const float* pressure, const float* rhs, float* out, int w, int h, int d) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || z >= d) return;

    const int id = idx3(x, y, z, w, h);
    if (x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1) {
        out[id] = 0.0f;
        return;
    }
    const float l = pressure[idx3(x - 1, y, z, w, h)];
    const float r = pressure[idx3(x + 1, y, z, w, h)];
    const float dn = pressure[idx3(x, y - 1, z, w, h)];
    const float up = pressure[idx3(x, y + 1, z, w, h)];
    const float bk = pressure[idx3(x, y, z - 1, w, h)];
    const float fr = pressure[idx3(x, y, z + 1, w, h)];
    const float lap = l + r + dn + up + bk + fr - 6.0f * pressure[id];
    out[id] = rhs[id] - lap;
}

/** Restrict: average the eight fine cells (2x2x2) into one coarse cell. */
__global__ void restrictKernel(const float* fine, float* coarse, int wf, int hf, int df) {
    const int xc = blockIdx.x * blockDim.x + threadIdx.x;
    const int yc = blockIdx.y * blockDim.y + threadIdx.y;
    const int zc = blockIdx.z * blockDim.z + threadIdx.z;
    const int wc = wf / 2;
    const int hc = hf / 2;
    const int dc = df / 2;
    if (xc >= wc || yc >= hc || zc >= dc) return;

    const int xf = xc * 2;
    const int yf = yc * 2;
    const int zf = zc * 2;
    float sum = 0.0f;
    #pragma unroll
    for (int dz = 0; dz < 2; ++dz)
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy)
    #pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
        sum += fine[idx3(xf + dx, yf + dy, zf + dz, wf, hf)];
    }
    coarse[idx3(xc, yc, zc, wc, hc)] = sum * 0.125f;
}

/** Prolong + add: trilinear interpolation of coarse onto fine, accumulated. */
__global__ void prolongAddKernel(const float* coarse, float* fine, int wf, int hf, int df) {
    const int xf = blockIdx.x * blockDim.x + threadIdx.x;
    const int yf = blockIdx.y * blockDim.y + threadIdx.y;
    const int zf = blockIdx.z * blockDim.z + threadIdx.z;
    if (xf >= wf || yf >= hf || zf >= df) return;

    const int wc = wf / 2;
    const int hc = hf / 2;
    const int dc = df / 2;

    const float cx = (xf - 0.5f) * 0.5f;
    const float cy = (yf - 0.5f) * 0.5f;
    const float cz = (zf - 0.5f) * 0.5f;

    int x0 = (int)floorf(cx);
    int y0 = (int)floorf(cy);
    int z0 = (int)floorf(cz);
    float tx = cx - (float)x0;
    float ty = cy - (float)y0;
    float tz = cz - (float)z0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    if (x0 < 0) { x0 = 0; tx = 0.0f; }
    if (y0 < 0) { y0 = 0; ty = 0.0f; }
    if (z0 < 0) { z0 = 0; tz = 0.0f; }
    if (x1 > wc - 1) x1 = wc - 1;
    if (y1 > hc - 1) y1 = hc - 1;
    if (z1 > dc - 1) z1 = dc - 1;
    if (x0 > wc - 1) x0 = wc - 1;
    if (y0 > hc - 1) y0 = hc - 1;
    if (z0 > dc - 1) z0 = dc - 1;

    const float c000 = coarse[idx3(x0, y0, z0, wc, hc)];
    const float c100 = coarse[idx3(x1, y0, z0, wc, hc)];
    const float c010 = coarse[idx3(x0, y1, z0, wc, hc)];
    const float c110 = coarse[idx3(x1, y1, z0, wc, hc)];
    const float c001 = coarse[idx3(x0, y0, z1, wc, hc)];
    const float c101 = coarse[idx3(x1, y0, z1, wc, hc)];
    const float c011 = coarse[idx3(x0, y1, z1, wc, hc)];
    const float c111 = coarse[idx3(x1, y1, z1, wc, hc)];

    const float c00 = c000 + tx * (c100 - c000);
    const float c10 = c010 + tx * (c110 - c010);
    const float c01 = c001 + tx * (c101 - c001);
    const float c11 = c011 + tx * (c111 - c011);
    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);
    const float val = c0 + tz * (c1 - c0);

    fine[idx3(xf, yf, zf, wf, hf)] += val;
}

}  // namespace

void smooth(float* pressure, const float* rhs, int w, int h, int d, int parity) {
    smoothKernel<<<grid3(w, h, d), dim3(kBlock, kBlock, kBlock)>>>(pressure, rhs, w, h, d, parity);
}

void residual(const float* pressure, const float* rhs, float* out, int w, int h, int d) {
    residualKernel<<<grid3(w, h, d), dim3(kBlock, kBlock, kBlock)>>>(pressure, rhs, out, w, h, d);
}

void restrict(const float* fine, float* coarse, int wf, int hf, int df) {
    const int wc = wf / 2, hc = hf / 2, dc = df / 2;
    restrictKernel<<<grid3(wc, hc, dc), dim3(kBlock, kBlock, kBlock)>>>(fine, coarse, wf, hf, df);
}

void prolongAdd(const float* coarse, float* fine, int wf, int hf, int df) {
    prolongAddKernel<<<grid3(wf, hf, df), dim3(kBlock, kBlock, kBlock)>>>(coarse, fine, wf, hf, df);
}

void zeroBuffer(float* buf, int count) {
    MG_CHECK(cudaMemset(buf, 0, sizeof(float) * (size_t)count));
}

void runVCycles(
    const std::vector<float*>& pressureLevels,
    const std::vector<float*>& rhsLevels,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& depths,
    int preSmooth,
    int postSmooth,
    int coarseIterations,
    int vcycles) {
    const int levels = (int)widths.size();

    for (int cycle = 0; cycle < vcycles; ++cycle) {
        // Down-sweep: smooth, compute residual, restrict to next level, zero coarse pressure.
        for (int L = 0; L < levels - 1; ++L) {
            for (int s = 0; s < preSmooth; ++s) {
                smooth(pressureLevels[L], rhsLevels[L], widths[L], heights[L], depths[L], 0);
                smooth(pressureLevels[L], rhsLevels[L], widths[L], heights[L], depths[L], 1);
            }
            // Residual is computed into a fine-sized scratch, then restricted into the
            // next-level rhs. Allocated locally to keep the public API simple.
            float* fineResidual = nullptr;
            const size_t bytes = sizeof(float) * (size_t)widths[L] * heights[L] * depths[L];
            MG_CHECK(cudaMalloc(&fineResidual, bytes));
            residual(pressureLevels[L], rhsLevels[L], fineResidual, widths[L], heights[L], depths[L]);
            restrict(fineResidual, rhsLevels[L + 1], widths[L], heights[L], depths[L]);
            MG_CHECK(cudaFree(fineResidual));
            zeroBuffer(pressureLevels[L + 1], widths[L + 1] * heights[L + 1] * depths[L + 1]);
        }

        // Coarsest solve: many GS sweeps.
        const int C = levels - 1;
        for (int s = 0; s < coarseIterations; ++s) {
            smooth(pressureLevels[C], rhsLevels[C], widths[C], heights[C], depths[C], 0);
            smooth(pressureLevels[C], rhsLevels[C], widths[C], heights[C], depths[C], 1);
        }

        // Up-sweep: prolong correction onto finer level, post-smooth.
        for (int L = levels - 2; L >= 0; --L) {
            prolongAdd(pressureLevels[L + 1], pressureLevels[L], widths[L], heights[L], depths[L]);
            for (int s = 0; s < postSmooth; ++s) {
                smooth(pressureLevels[L], rhsLevels[L], widths[L], heights[L], depths[L], 0);
                smooth(pressureLevels[L], rhsLevels[L], widths[L], heights[L], depths[L], 1);
            }
        }
    }
}

}  // namespace mg
