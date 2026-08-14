#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <kompute/Kompute.hpp>
#include <vulkan/vulkan.h>
#include <cstring>

class UniformBufferHelper {
public:
  UniformBufferHelper(std::shared_ptr<kp::Manager> mgr, VkDeviceSize size)
    : mMgr(mgr), mSize(size) {
    createUniformBuffer();
  }

  ~UniformBufferHelper() {
    if (mVkBuffer) {
      mVkDevice->destroyBuffer(mVkBuffer);
    }
    if (mVkDeviceMemory) {
      mVkDevice->freeMemory(mVkDeviceMemory);
    }
  }

  // Update UBO contents in algorithm
  template <typename T>
  void setData(const T& data, std::shared_ptr<kp::Algorithm> algo, uint32_t bind_location) {
    void *mapped = mVkDevice->mapMemory(mVkDeviceMemory, 0, sizeof(T));
    std::memcpy(mapped, &data, sizeof(T));
    mVkDevice->unmapMemory(mVkDeviceMemory);
    vk::DescriptorBufferInfo uboInfo = descriptorVkInfo();
    vk::DescriptorSet algDescriptorSet = *algo->getDescriptorSet().get();
    std::vector<vk::WriteDescriptorSet> writes = { vk::WriteDescriptorSet(
        algDescriptorSet,
        bind_location,
        0,
        1,
        vk::DescriptorType::eUniformBuffer,
        nullptr, // no image info
        &uboInfo, // buffer info
        nullptr
    ) };

    mMgr->getDevice()->updateDescriptorSets(writes, nullptr);
  }

  template <typename T>
  void modifyData(const T& data, std::shared_ptr<kp::Algorithm> algo, uint32_t bind_location) {
    void *mapped = mVkDevice->mapMemory(mVkDeviceMemory, 0, sizeof(T));
    std::memcpy(mapped, &data, sizeof(T));
    mVkDevice->unmapMemory(mVkDeviceMemory);
  }

  // Return VkDescriptorBufferInfo for binding manually or via Kompute algorithm
  vk::DescriptorBufferInfo descriptorVkInfo() const {
    vk::DescriptorBufferInfo info{};
    info.buffer = mVkBuffer;
    info.offset = 0;
    info.range = mSize;
    return info;
  }

  vk::Buffer getBuffer() const { return mVkBuffer; }
  vk::DeviceMemory getMemory() const { return mVkDeviceMemory; }
  vk::DeviceSize getSize() const { return mSize; }

private:
  std::shared_ptr<kp::Manager> mMgr;
  std::shared_ptr<vk::Device> mVkDevice;
  std::shared_ptr<vk::PhysicalDevice> mVkPhysicalDevice;
  vk::Buffer mVkBuffer;
  vk::DeviceMemory mVkDeviceMemory;
  vk::DeviceSize mSize = 0;
  const vk::MemoryPropertyFlags mMemPropsDesired = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

  void createUniformBuffer() {
    mVkDevice = mMgr->getDevice();
    mVkPhysicalDevice = mMgr->getPhysicalDevice();

    vk::BufferCreateInfo bufferInfo{
        vk::BufferCreateFlags(),
        mSize,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::SharingMode::eExclusive // the default
    };
  
    // vk::Result res = mVkDevice->createBuffer(&bufferInfo, nullptr, &mVkBuffer);
    mVkBuffer = mVkDevice->createBuffer(bufferInfo, nullptr, VULKAN_HPP_DISPATCH_LOADER_DYNAMIC);
    //if (res != vk::Result::eSuccess) {
    //    throw std::runtime_error("Failed to create UBO buffer!");
    //}

    vk::MemoryRequirements vk_memReq = mVkDevice->getBufferMemoryRequirements(mVkBuffer);

    vk::PhysicalDeviceMemoryProperties vk_memProps = mVkPhysicalDevice->getMemoryProperties();

    uint32_t memTypeIndex = findMemoryTypeIndex(
      *(mVkPhysicalDevice.get()),
      vk_memReq.memoryTypeBits,
      mMemPropsDesired
    );

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = vk_memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = nullptr;

    mVkDeviceMemory = mVkDevice->allocateMemory(allocInfo);
    mVkDevice->bindBufferMemory(mVkBuffer, mVkDeviceMemory, 0);
  }

  static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties,
    const VkPhysicalDeviceMemoryProperties& memProps) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    throw std::runtime_error("Failed to find suitable memory type for UBO");
  }

  uint32_t findMemoryTypeIndex(
    const vk::PhysicalDevice& physicalDevice,
    uint32_t typeFilter,
    vk::MemoryPropertyFlags properties) {
    auto memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & 1) == 1) {
        if ((memProps.memoryTypes[i].propertyFlags & properties) ==
          properties) {
          return i;
        }
      }
      typeFilter >>= 1;
    }
    return 0;
  }
};
