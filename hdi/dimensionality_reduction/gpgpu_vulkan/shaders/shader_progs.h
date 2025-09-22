#pragma once
#include "shaders.h"
#include "../tensor_config.h"
#include <kompute/Kompute.hpp>

class BoundsShaderProg {
public:
  BoundsShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }

  std::vector<float> compute(float padding);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::BOUNDS] };
  kp::Manager& _mgr;
  TensorMap& _tensors;

};

class StencilShaderProg {
public:
  StencilShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  std::vector<uint8_t> compute(uint32_t width, uint32_t height, unsigned int num_points, std::vector<float> bounds);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::STENCIL] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class FieldComputationShaderProg {
public:
  FieldComputationShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  std::vector<float> compute(std::vector<uint8_t> stencil, uint32_t width, uint32_t height);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::COMPUTE_FIELDS] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
  const float _function_support = 6.5f;
};

class InterpolationShaderProg {
public:
  InterpolationShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  void compute(std::vector<float> fields, uint32_t width, uint32_t height);
  float getSumQ() { return _sum_Q; };
private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::INTERP_FIELDS] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
  float _sum_Q = 0.0f;
};

class ForcesShaderProg {
public:
  ForcesShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  float compute(unsigned int num_points, float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::COMPUTE_FORCES] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class UpdateShaderProg {
public:
  UpdateShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  void compute(unsigned int num_points, float eta, float minimum_gain, float iteration, float momentumn, unsigned int momentum_switch, float final_momentum, float gain_mult);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::UPDATE] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
};

class CenterScaleShaderProg {
public:
  CenterScaleShaderProg(kp::Manager& mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors) {
  }
  void compute(unsigned int num_points, float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary{ SPIRVShaderBinaries[SPIRVShader::CENTER_SCALE] };
  kp::Manager& _mgr;
  TensorMap& _tensors;
};
