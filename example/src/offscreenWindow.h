#pragma once

struct GLFWwindow;

class OffscreenBuffer
{
public:
    OffscreenBuffer() : _isInitialized(false) {}

    bool isInitialized() const { return _isInitialized; }

    /** Initialize and bind the OpenGL context associated with this buffer */
    virtual void initialize() = 0;

    /** Bind the OpenGL context associated with this buffer */
    virtual void bindContext() = 0;

    /** Release the OpenGL context associated with this buffer */
    virtual void releaseContext() = 0;

    /** Destroy the OpenGL context associated with this buffer */
    virtual void destroyContext() = 0;

protected:
    bool _isInitialized;

};

class OffscreenBufferGLFW : public OffscreenBuffer
{
public:
    OffscreenBufferGLFW() = default;

    void initialize() override;
    void bindContext() override;
    void releaseContext() override;
    void destroyContext() override;

private:
    GLFWwindow* _offscreenWindow = nullptr;
};
