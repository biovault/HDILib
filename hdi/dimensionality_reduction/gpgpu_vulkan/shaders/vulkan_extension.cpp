#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/vulkan_extension.h"

// record() will be called by kp::Sequence with a vk::CommandBuffer
void OpIndirectDispatch::record(const vk::CommandBuffer& commandBuffer) {
  ;
  for (auto& mem : mAlgorithm->getMemObjects()) {
    // Only for images
    if (auto img = dynamic_cast<kp::Image*>(mem.get())) {
      img->recordPrimaryImageBarrier(
        commandBuffer,
        vk::AccessFlagBits::eTransferWrite,              // previous writes
        vk::AccessFlagBits::eShaderWrite,
        vk::PipelineStageFlagBits::eTransfer,            // or TOP_OF_PIPE if new
        vk::PipelineStageFlagBits::eComputeShader,       // shader stage that uses it
        vk::ImageLayout::eGeneral
      );
    }
  }
  // Ensure the indirect buffer is visible to the indirect-read stage.
  // (use access/stage masks appropriate for indirect commands)
  mDispatchTensor->recordPrimaryMemoryBarrier(
    commandBuffer,
    vk::AccessFlagBits::eTransferWrite,          // srcAccessMask
    vk::AccessFlagBits::eIndirectCommandRead,// dstAccessMask
    vk::PipelineStageFlagBits::eTransfer,       // srcStageMask
    vk::PipelineStageFlagBits::eDrawIndirect// dstStageMask (indirect read)
  );

  // Bind the algorithm's pipeline + descriptor sets
  mAlgorithm->recordBindCore(commandBuffer);

  // If you need the algorithm's push constants overridden, use:
  mAlgorithm->recordBindPush(commandBuffer);

  // get the underlying vk::Buffer (DescriptorBufferInfo contains the buffer)
  auto desc = mDispatchTensor->constructDescriptorBufferInfo();

  // issue the indirect dispatch using Vulkan-Hpp
  commandBuffer.dispatchIndirect(desc.buffer, 0);
}

