#pragma once
#include "shaders.h"
#include "../tensor_config.h"
#include <kompute/Kompute.hpp>

class BoundsShaderProg {
public:
  BoundsShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::BOUNDS] )
  {}

  std::vector<float> compute(float padding = 0.0);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class StencilShaderProg {
public:
  StencilShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors), 
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::STENCIL]) {
  }
  std::shared_ptr<kp::ImageT<float>> compute(uint32_t width, uint32_t height, unsigned int num_points, std::vector<float> bounds);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class FieldComputationShaderProg {
public:
  FieldComputationShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FIELDS])
  {
  }
  std::shared_ptr<kp::ImageT<float>> compute(std::shared_ptr<kp::ImageT<float>> stencil, uint32_t width, uint32_t height);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
  const float _function_support = 6.5f;
};

class InterpolationShaderProg {
public:
  InterpolationShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::INTERP_FIELDS])
  {
  }
  void compute(std::shared_ptr<kp::ImageT<float>> fields, uint32_t width, uint32_t height);
  float getSumQ() { return _sum_Q; };
private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
  float _sum_Q = 0.0f;
};

class ForcesShaderProg {
public:
  ForcesShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FORCES])
  {
  }
  float compute(unsigned int num_points, float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class UpdateShaderProg {
public:
  UpdateShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::UPDATE])
  {
  }
  void compute(unsigned int num_points, float eta, float minimum_gain, float iteration, float momentumn, unsigned int momentum_switch, float final_momentum, float gain_mult);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class CenterScaleShaderProg {
public:
  CenterScaleShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::CENTER_SCALE])
  {
  }
  std::vector<float> compute(unsigned int num_points, float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  kp::Manager& _mgr;
  TensorMap& _tensors;
};
