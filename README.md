# CUDA Fluids

## Situation

Realtime fluid simulations need a tight loop between GPU computation, user input, and visual feedback. This project is a Windows-focused CUDA implementation of a 3D dye, temperature, and velocity simulation displayed in a GLFW/OpenGL window via a max-intensity-projection along z.

## Task

Build a 3D fluid solver that can:

- Store the full 3D simulation state on the GPU.
- Advect velocity, density, and temperature with semi-Lagrangian (trilinear) sampling.
- Use temperature for buoyant rising motion.
- Preserve visible swirl with full 3D vortex confinement.
- Apply viscous diffusion via implicit Jacobi iteration.
- Solve the pressure Poisson equation with a geometric multigrid V-cycle.
- Display the volume in realtime (max-intensity projection).
- Accept mouse input for interactive dye, heat, and velocity impulses at the mid-z slice.
- Build reliably from normal PowerShell on Windows with MSVC, Ninja, CMake, and NVCC.

## Action

The implementation uses:

- `include/Fluid3D.cuh` for the solver interface and runtime settings (`FluidSettings`).
- `src/Fluid3D.cu` for CUDA kernels: 3D semi-Lagrangian advection, buoyancy, 3D vorticity + confinement, implicit-Jacobi viscous diffusion, divergence, projection, impulse injection.
- `include/Multigrid.cuh` / `src/Multigrid.cu` for the geometric multigrid V-cycle pressure solver — red-black Gauss-Seidel smoother, full-weighting restriction, trilinear prolongation.
- `include/VolumeRender.cuh` / `src/VolumeRender.cu` for a CUDA max-intensity-projection kernel that renders the 3D volumes into a 2D RGBA8 image per frame.
- `src/main.cpp` for the GLFW/OpenGL window, fixed-timestep loop, mouse input, and texture upload.
- `scripts/build-msvc.ps1` for entering the x64 Visual Studio developer environment before configuring and building.
- `CMakeLists.txt` for CUDA, OpenGL, and fetched GLFW 3.4 build configuration.

### Build

From normal PowerShell:

```powershell
.\scripts\build-msvc.ps1 -BuildDir build
```

To specify a CUDA architecture explicitly:

```powershell
.\scripts\build-msvc.ps1 -BuildDir build -CudaArchitectures 86
```

Manual build from an x64 Visual Studio developer shell:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

### Run

```powershell
.\build\cuda_fluids.exe --width 64 --height 64 --depth 64 --window-scale 8
```

Grid dimensions must each be divisible by `2^(mg-levels - 1)` so the multigrid hierarchy can halve cleanly. With the default of 4 levels that is a multiple of 8.

Tuning examples:

```powershell
.\build\cuda_fluids.exe --vorticity-strength 24
.\build\cuda_fluids.exe --buoyancy-strength 48
.\build\cuda_fluids.exe --viscosity 1e-2
.\build\cuda_fluids.exe --mg-levels 5 --mg-vcycles 1
```

Quick smoke test:

```powershell
.\build\cuda_fluids.exe --width 32 --height 32 --depth 32 --max-frames 10
```

Controls:

- Left-drag injects density, heat, and velocity at the mid-z slice.
- `R` resets the simulation.
- `Esc` exits.

## Result

A realtime interactive 3D CUDA fluid solver: semi-Lagrangian advection, viscous diffusion, vortex confinement, buoyancy from temperature, and a geometric multigrid pressure projection — all rendered as a max-intensity projection in a GLFW window.
