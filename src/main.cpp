#include "Fluid3D.cuh"
#include "VolumeRender.cuh"

/**
 * @file main.cpp
 * @brief Realtime GLFW/OpenGL front end for the CUDA 3D fluid simulation.
 *
 * The window shows a max-intensity projection of the 3D density and temperature
 * volumes along the z axis. Mouse drags inject dye, heat, and velocity at the
 * mid-z slice.
 */

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AppOptions {
    int width = 64;
    int height = 64;
    int depth = 64;
    int windowScale = 8;
    int maxFrames = 0;
    float buoyancyStrength = 24.0f;
    float vorticityStrength = 12.0f;
    float viscosity = 1.0e-4f;
    int mgLevels = 4;
    int mgVCycles = 2;
};

struct MouseState {
    bool hasPrevious = false;
    double previousX = 0.0;
    double previousY = 0.0;
};

int parseInt(const char* text, const std::string& name) {
    try {
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != std::string(text).size()) throw std::invalid_argument("trailing");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer for " + name + ": " + text);
    }
}

float parseFloat(const char* text, const std::string& name) {
    try {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != std::string(text).size()) throw std::invalid_argument("trailing");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid float for " + name + ": " + text);
    }
}

AppOptions parseArgs(int argc, char** argv) {
    AppOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto val = [&](const std::string& n) -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value after " + n);
            return argv[++i];
        };
        if (arg == "--width") o.width = parseInt(val(arg), arg);
        else if (arg == "--height") o.height = parseInt(val(arg), arg);
        else if (arg == "--depth") o.depth = parseInt(val(arg), arg);
        else if (arg == "--window-scale") o.windowScale = parseInt(val(arg), arg);
        else if (arg == "--max-frames") o.maxFrames = parseInt(val(arg), arg);
        else if (arg == "--buoyancy-strength") o.buoyancyStrength = parseFloat(val(arg), arg);
        else if (arg == "--vorticity-strength") o.vorticityStrength = parseFloat(val(arg), arg);
        else if (arg == "--viscosity") o.viscosity = parseFloat(val(arg), arg);
        else if (arg == "--mg-levels") o.mgLevels = parseInt(val(arg), arg);
        else if (arg == "--mg-vcycles") o.mgVCycles = parseInt(val(arg), arg);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cuda_fluids [options]\n"
                << "  --width N --height N --depth N   Grid dimensions, default 64\n"
                << "  --window-scale N                 Initial window scale, default 8\n"
                << "  --max-frames N                   Exit after N frames, default 0 = unlimited\n"
                << "  --buoyancy-strength F            Default 24\n"
                << "  --vorticity-strength F           Default 12\n"
                << "  --viscosity F                    Kinematic viscosity, default 1e-4\n"
                << "  --mg-levels N                    Multigrid levels, default 4\n"
                << "  --mg-vcycles N                   V-cycles per step, default 2\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    if (o.width < 4 || o.height < 4 || o.depth < 4) {
        throw std::invalid_argument("Grid dimensions must be at least 4.");
    }
    return o;
}

void requireGlfw(bool ok, const std::string& msg) {
    if (!ok) {
        const char* desc = nullptr;
        glfwGetError(&desc);
        throw std::runtime_error(msg + (desc ? ": " + std::string(desc) : ""));
    }
}

void renderTexture(GLuint texture, int fbW, int fbH) {
    glViewport(0, 0, fbW, fbH);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
}

void addMouseImpulse(GLFWwindow* window, Fluid3D& fluid, MouseState& mouse) {
    const bool down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (!down) { mouse.hasPrevious = false; return; }

    int ww = 0, wh = 0;
    glfwGetWindowSize(window, &ww, &wh);
    if (ww <= 0 || wh <= 0) return;

    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(window, &cx, &cy);
    const float nx = std::clamp(static_cast<float>(cx / (double)ww), 0.0f, 1.0f);
    const float ny = std::clamp(1.0f - static_cast<float>(cy / (double)wh), 0.0f, 1.0f);
    const float nz = 0.5f;

    float3 vel = make_float3(0, 0, 0);
    if (mouse.hasPrevious) {
        const auto& s = fluid.settings();
        const float dx = static_cast<float>((cx - mouse.previousX) / (double)ww);
        const float dy = static_cast<float>((mouse.previousY - cy) / (double)wh);
        vel = make_float3(dx * (float)s.width * 24.0f, dy * (float)s.height * 24.0f, 0.0f);
    }

    fluid.addImpulse(nx, ny, nz, 0.06f, 3.5f, 2.25f, vel);
    mouse.previousX = cx;
    mouse.previousY = cy;
    mouse.hasPrevious = true;
}

void addDemoImpulse(Fluid3D& fluid, double time) {
    const float sx = 0.5f + 0.18f * std::sin(static_cast<float>(time) * 1.6f);
    const float sy = 0.25f;
    const float sz = 0.5f + 0.18f * std::cos(static_cast<float>(time) * 1.1f);
    const float swirlX = 18.0f * std::cos(static_cast<float>(time) * 2.3f);
    const float swirlZ = 18.0f * std::sin(static_cast<float>(time) * 1.7f);
    fluid.addImpulse(sx, sy, sz, 0.05f, 0.85f, 1.35f, make_float3(swirlX, 0.0f, swirlZ));
}

GLuint createFluidTexture(int w, int h) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    std::vector<unsigned char> empty((size_t)w * h * 4, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, empty.data());
    return t;
}

void updateWindowTitle(GLFWwindow* window, double elapsed, int frames) {
    if (elapsed <= 0.0) return;
    std::ostringstream t;
    t << "CUDA Fluids 3D | " << std::fixed << std::setprecision(1)
      << ((double)frames / elapsed) << " FPS";
    glfwSetWindowTitle(window, t.str().c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppOptions options = parseArgs(argc, argv);

        FluidSettings settings;
        settings.width = options.width;
        settings.height = options.height;
        settings.depth = options.depth;
        settings.buoyancyStrength = options.buoyancyStrength;
        settings.vorticityStrength = options.vorticityStrength;
        settings.viscosity = options.viscosity;
        settings.mgLevels = options.mgLevels;
        settings.mgVCycles = options.mgVCycles;

        requireGlfw(glfwInit() == GLFW_TRUE, "Failed to initialize GLFW");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

        GLFWwindow* window = glfwCreateWindow(
            options.width * options.windowScale,
            options.height * options.windowScale,
            "CUDA Fluids 3D", nullptr, nullptr);
        requireGlfw(window != nullptr, "Failed to create GLFW window");

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        Fluid3D fluid(settings);
        volr::Renderer renderer(settings.width, settings.height);
        MouseState mouse;
        GLuint fluidTexture = createFluidTexture(settings.width, settings.height);
        std::vector<unsigned char> pixels((size_t)settings.width * settings.height * 4, 0);

        bool resetWasPressed = false;
        double lastTime = glfwGetTime();
        double accumulator = settings.dt;
        double titleTimer = lastTime;
        int titleFrameCount = 0;
        int totalFrameCount = 0;

        while (!glfwWindowShouldClose(window)) {
            const double now = glfwGetTime();
            const double frameTime = std::min(now - lastTime, 0.1);
            lastTime = now;
            accumulator += frameTime;

            glfwPollEvents();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            const bool resetIsPressed = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
            if (resetIsPressed && !resetWasPressed) fluid.reset();
            resetWasPressed = resetIsPressed;

            while (accumulator >= settings.dt) {
                addDemoImpulse(fluid, now);
                addMouseImpulse(window, fluid, mouse);
                fluid.step();
                accumulator -= settings.dt;
            }

            int fbW = 0, fbH = 0;
            glfwGetFramebufferSize(window, &fbW, &fbH);
            if (fbW > 0 && fbH > 0) {
                renderer.render(
                    fluid.densityDevice(), fluid.temperatureDevice(),
                    settings.width, settings.height, settings.depth,
                    0.36f, 0.42f);
                renderer.copyToHost(pixels.data());
                glBindTexture(GL_TEXTURE_2D, fluidTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    settings.width, settings.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                renderTexture(fluidTexture, fbW, fbH);
                glfwSwapBuffers(window);
                ++titleFrameCount;
                ++totalFrameCount;
            }

            if (now - titleTimer >= 0.5) {
                updateWindowTitle(window, now - titleTimer, titleFrameCount);
                titleTimer = now;
                titleFrameCount = 0;
            }
            if (options.maxFrames > 0 && totalFrameCount >= options.maxFrames) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }

        glDeleteTextures(1, &fluidTexture);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        glfwTerminate();
        return 1;
    }
}
