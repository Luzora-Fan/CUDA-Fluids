#pragma once

/**
 * @file Multigrid.cuh
 * @brief Geometric multigrid V-cycle for the 3D pressure Poisson equation.
 *
 * Solves (sum_neighbors - 6 * p) = rhs (unit grid spacing) with Dirichlet
 * pressure = 0 at the boundary. Each level halves width, height, and depth.
 */

#include <cuda_runtime.h>

#include <vector>

namespace mg {

/** One red-black GS sweep over half the cells. parity = 0 (red) or 1 (black). */
void smooth(float* pressure, const float* rhs, int w, int h, int d, int parity);

/** Computes residual = rhs - (sum_neighbors - 6 * pressure). */
void residual(const float* pressure, const float* rhs, float* out, int w, int h, int d);

/** Full-weighting restriction from a fine field to a coarse field of half the dimensions. */
void restrict(const float* fine, float* coarse, int wf, int hf, int df);

/** Trilinear prolongation that adds the coarse correction onto the fine pressure. */
void prolongAdd(const float* coarse, float* fine, int wf, int hf, int df);

/** Zero a device buffer of given element count (float). */
void zeroBuffer(float* buf, int count);

/**
 * @brief Runs `vcycles` V-cycles on the supplied pre-allocated multigrid hierarchy.
 *
 * @param pressureLevels  Per-level pressure buffers; level 0 is finest. Level 0
 *                        is updated in place. Coarser levels are scratch.
 * @param rhsLevels       Per-level right-hand side buffers. Level 0 must hold
 *                        the divergence on entry; coarser levels are scratch.
 * @param widths/heights/depths  Per-level dimensions, finest first.
 */
void runVCycles(
    const std::vector<float*>& pressureLevels,
    const std::vector<float*>& rhsLevels,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& depths,
    int preSmooth,
    int postSmooth,
    int coarseIterations,
    int vcycles);

}  // namespace mg
