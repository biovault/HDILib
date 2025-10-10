#pragma once

#include <kompute/Kompute.hpp>
#include <vulkan/vulkan.hpp>

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