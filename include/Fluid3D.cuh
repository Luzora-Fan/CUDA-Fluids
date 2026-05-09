#pragma once

/**
 * @file Fluid3D.cuh
 * @brief Public interface for the CUDA-backed 3D fluid simulation.
 */

#include <cuda_runtime.h>

#include <vector>

/**
 * @brief Runtime parameters that control grid size and solver behavior.
 *
 * Fields are stored on a collocated grid: one velocity vector, density,
 * temperature, pressure, divergence, and vorticity vector per cell. Width,
 * height, and depth must each be divisible by 2^(mgLevels - 1) so the
 * geometric multigrid V-cycle can halve each axis cleanly.
 */
struct FluidSettings {
    int width = 64;
    int height = 64;
    int depth = 64;

    float dt = 1.0f / 60.0f;

    float velocityDissipation = 0.999f;
    float densityDissipation = 0.998f;
    float temperatureDissipation = 0.996f;

    float buoyancyStrength = 24.0f;
    float vorticityStrength = 12.0f;

    /** Kinematic viscosity used by the implicit Jacobi diffusion step. */
    float viscosity = 1.0e-4f;
    int viscosityIterations = 20;

    /** Geometric multigrid V-cycle parameters. */
    int mgLevels = 4;
    int mgPreSmooth = 2;
    int mgPostSmooth = 2;
    int mgVCycles = 2;
    /** Number of red-black GS sweeps on the coarsest level per V-cycle. */
    int mgCoarseIterations = 32;
};

/**
 * @brief Owns GPU buffers and advances an incompressible 3D fluid.
 *
 * Pipeline per step: impulse, semi-Lagrangian velocity advection, buoyancy,
 * vortex confinement, viscosity (implicit Jacobi), divergence, multigrid
 * pressure projection, scalar advection (density, temperature).
 */
class Fluid3D {
public:
    explicit Fluid3D(FluidSettings settings);
    ~Fluid3D();

    Fluid3D(const Fluid3D&) = delete;
    Fluid3D& operator=(const Fluid3D&) = delete;
    Fluid3D(Fluid3D&&) = delete;
    Fluid3D& operator=(Fluid3D&&) = delete;

    void reset();

    /** Sphere-falloff impulse at a normalized window-space position; depth fixed at mid-z. */
    void addImpulse(
        float normalizedX,
        float normalizedY,
        float normalizedZ,
        float normalizedRadius,
        float densityAmount,
        float temperatureAmount,
        float3 velocityAmount);

    void step();

    std::vector<float> copyDensityToHost() const;
    std::vector<float> copyTemperatureToHost() const;

    /** GPU pointers for the volume renderer (read-only use). */
    const float* densityDevice() const { return density_; }
    const float* temperatureDevice() const { return temperature_; }

    const FluidSettings& settings() const { return settings_; }

private:
    int cellCount() const;
    size_t scalarBytes() const;
    size_t vectorBytes() const;

    FluidSettings settings_{};
    float3* velocity_ = nullptr;
    float3* velocityScratch_ = nullptr;
    float* density_ = nullptr;
    float* densityScratch_ = nullptr;
    float* temperature_ = nullptr;
    float* temperatureScratch_ = nullptr;
    float* pressure_ = nullptr;
    float* divergence_ = nullptr;
    float3* vorticity_ = nullptr;

    /** Multigrid hierarchy buffers (level 0 is finest). pressureLevels_[0] aliases pressure_. */
    std::vector<float*> mgPressure_;
    std::vector<float*> mgRhs_;
    std::vector<int> mgWidths_;
    std::vector<int> mgHeights_;
    std::vector<int> mgDepths_;
};
