#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_metal.h>

void* getMetalDevice(vk::PhysicalDevice phys)
{
    VkPhysicalDeviceMetalFeaturesMVK features{};
    features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_METAL_FEATURES_MVK;

    vkGetPhysicalDeviceMetalFeaturesMVK(
        static_cast<VkPhysicalDevice>(phys),
        &features);

    return features.metalDevice; // id<MTLDevice>
}
