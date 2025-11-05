#pragma once

#include <kompute/Kompute.hpp>
#include <vulkan/vulkan.hpp>

/**
 * @brief OpIndirectDispatch is an operation that performs an indirect dispatch
 * using a dispatch parameters tensor.
 */
class OpIndirectDispatch : public kp::OpBase {
public:
  OpIndirectDispatch(std::shared_ptr<kp::Algorithm> algorithm,
    std::shared_ptr<kp::Tensor> dispatchTensor)
    : mAlgorithm(std::move(algorithm))
    , mDispatchTensor(std::move(dispatchTensor)) {
  }

  void record(const vk::CommandBuffer& commandBuffer);
  void preEval(const vk::CommandBuffer& commandBuffer) override {}
  void postEval(const vk::CommandBuffer& commandBuffer) override {}

private:
  std::shared_ptr<kp::Algorithm> mAlgorithm;
  std::shared_ptr<kp::Tensor>    mDispatchTensor;
};

/**
 * @brief OpImageLayoutTransition is an operation that performs an
 * image layout transition. Primary intended use is to transition images used in
 * compute shaders between general and shader read optimal layouts. 
 */
class OpImageLayoutTransition : public kp::OpBase {
public:
  OpImageLayoutTransition(
    std::shared_ptr<kp::ImageT<float>> img,
    vk::ImageLayout oldL,
    vk::ImageLayout newL)
    : mImage(std::move(img)), mOldLayout(oldL), mNewLayout(newL) {
  }

  void record(const vk::CommandBuffer& commandBuffer) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.sType = vk::StructureType::eImageMemoryBarrier2;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = this->mOldLayout;
    barrier.newLayout = this->mNewLayout;
    barrier.image = *this->mImage->getPrimaryImage().get();
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::DependencyInfo depInfo{};
    depInfo.sType = vk::StructureType::eDependencyInfo;
    depInfo.dependencyFlags = {};
    depInfo.memoryBarrierCount = 0;
    depInfo.pMemoryBarriers = nullptr;
    depInfo.bufferMemoryBarrierCount = 0;
    depInfo.pBufferMemoryBarriers = nullptr;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;

    commandBuffer.pipelineBarrier2(depInfo);
  }
  void preEval(const vk::CommandBuffer& commandBuffer) override {}
  void postEval(const vk::CommandBuffer& commandBuffer) override {}

private:
  std::shared_ptr<kp::ImageT<float>> mImage;
  vk::ImageLayout mOldLayout;
  vk::ImageLayout mNewLayout;
};
