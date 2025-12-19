#include "MetalDevice.h"


// From the active logical vulkan device return the underlying Metal device 
// This code is APPLE only
MTLDevice_id getMetalDevice(vk::Device device)
{
    VkExportMetalDeviceInfoEXT exportDeviceInfo{
        VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT,
        nullptr,
        nullptr
    };

    VkExportMetalObjectsInfoEXT exportObjectsInfo{
        VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
        &exportDeviceInfo
    };

    auto fp_vkExportMetalObjectsEXT =
    reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        device.getProcAddr("vkExportMetalObjectsEXT"));
    
    if (!fp_vkExportMetalObjectsEXT) {
        throw std::runtime_error("vkExportMetalObjectsEXT not available");
    }

    fp_vkExportMetalObjectsEXT(
        static_cast<VkDevice>(device),
        &exportObjectsInfo
    );
    
    return exportDeviceInfo.mtlDevice;

}
