#include "offscreenWindow.h"

#include <hdi/utils/glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>

void OffscreenBufferGLFW::initialize()
{
    if (!glfwInit()) {
        throw std::runtime_error("Unable to initialize GLFW.");
    }

#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // invisible - ie offscreen, window
    _offscreenWindow = glfwCreateWindow(640, 480, "", NULL, NULL);

    if (_offscreenWindow == NULL) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    bindContext();

#ifndef __APPLE__
    if (!gladLoadGL(glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize OpenGL context");
    }
#endif // Not __APPLE__

    _isInitialized = true;
}

void OffscreenBufferGLFW::bindContext()
{
    glfwMakeContextCurrent(_offscreenWindow);
}

void OffscreenBufferGLFW::releaseContext()
{
    glfwMakeContextCurrent(nullptr);
}

void OffscreenBufferGLFW::destroyContext()
{
    releaseContext();
    glfwDestroyWindow(_offscreenWindow);
    glfwTerminate();
    _isInitialized = false;
}

