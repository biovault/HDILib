#pragma once

struct GLFWwindow;

class OffscreenBuffer
{
public:
    OffscreenBuffer() = default;
    virtual ~OffscreenBuffer() = default;

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
    bool _isInitialized = false;

};

class OffscreenBufferGLFW : public OffscreenBuffer
{
public:
    OffscreenBufferGLFW() = default;
    ~OffscreenBufferGLFW() override;

    OffscreenBufferGLFW(const OffscreenBufferGLFW&) = delete;
    OffscreenBufferGLFW& operator=(const OffscreenBufferGLFW&) = delete;

    OffscreenBufferGLFW(OffscreenBufferGLFW&& other) = delete;
    OffscreenBufferGLFW& operator=(OffscreenBufferGLFW&& other) = delete;

    void initialize() override;
    void bindContext() override;
    void releaseContext() override;
    void destroyContext() override;

private:
    GLFWwindow* _offscreenWindow = nullptr;
};
