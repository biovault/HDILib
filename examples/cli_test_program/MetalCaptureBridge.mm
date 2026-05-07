#import <Metal/Metal.h>

extern "C" {

void StartMetalCapture(void* metalDevice)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)metalDevice;

    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = device;

    NSError* error = nil;
    [[MTLCaptureManager sharedCaptureManager]
        startCaptureWithDescriptor:desc
        error:&error];
}

void EndMetalCapture()
{
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

}