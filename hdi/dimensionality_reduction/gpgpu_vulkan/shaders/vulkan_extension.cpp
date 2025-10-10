#include "vulkan_extension.h"

// record() will be called by kp::Sequence with a vk::CommandBuffer
void OpIndirectDispatch::record(const vk::CommandBuffer& commandBuffer) {
  // Ensure the indirect buffer is visible to the indirect-read stage.
  // (use access/stage masks appropriate for indirect commands)
  mDispatchTensor->recordPrimaryMemoryBarrier(
    commandBuffer,
    vk::AccessFlagBits::eHostWrite,          // srcAccessMask
    vk::AccessFlagBits::eIndirectCommandRead,// dstAccessMask
    vk::PipelineStageFlagBits::eHost,       // srcStageMask
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

