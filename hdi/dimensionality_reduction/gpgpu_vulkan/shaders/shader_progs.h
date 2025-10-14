#pragma once
#include "shaders.h"
#include "../tensor_config.h"
#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>
#include "vkUniformBufferHelper.h"


// The shader programs present two separate APIS:
// 1) compute() function that runs the shader program immediately and returns the result
// 2) record() and update() function that records the commands into a kompute::Sequence for later execution
// Do not mix the two APIs for the same shader program instance (or bad things will happen)
// API 1) is useful for development and debugging, while API 2) is more efficient for production use cases
class BoundsShaderProg {
public:
  BoundsShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::BOUNDS] )
  {}

  std::vector<float> compute(float padding = 0.0);

  void record_padded(
    std::shared_ptr<kp::Sequence> seq,
    float padding = 0.1);

  void record_unpadded(
    std::shared_ptr<kp::Sequence> seq);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _boundsAlgorithmPadded;
  std::shared_ptr<kp::Algorithm> _boundsAlgorithmUnpadded;
  TensorMap& _tensors;
};

class StencilShaderProg {
public:
  StencilShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::STENCIL]),
    _fields_buffer_size(0),
    _stencil_array(std::vector<float>(0)),
    _ubo(mgr, sizeof(stencilParams))
  {
  };

  std::shared_ptr<kp::ImageT<float>> compute(
    uint32_t width, uint32_t height, 
    unsigned int num_points, 
    std::vector<float> bounds);

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t width, uint32_t height,
    unsigned int num_points,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t width, uint32_t height,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);

  std::shared_ptr<kp::ImageT<float>> getStencil() const {
    return _stencil_out;
  };

  std::shared_ptr<kp::ImageT<float>> _stencil_out;
private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _stencilAlgorithm;
  TensorMap& _tensors;
  unsigned int _fields_buffer_size;
  std::vector<float> _stencil_array;
  UniformBufferHelper _ubo;

};

class FieldComputationShaderProg {
public:
  FieldComputationShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FIELDS]),
    _fields_buffer_size(0),
    _field_array(std::vector<float>(0))
  {
  };

  std::shared_ptr<kp::ImageT<float>> compute(
    std::shared_ptr<kp::ImageT<float>> stencil, 
    uint32_t width, uint32_t height);

  void record(
    std::shared_ptr<kp::Sequence> seq,
    std::shared_ptr<kp::ImageT<float>> stencil,
    uint32_t width, uint32_t height,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t width, uint32_t height,
    unsigned int new_fields_buffer_size = 0);

  std::shared_ptr<kp::ImageT<float>> _field_out;
private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _fieldAlgorithm;
  TensorMap& _tensors;
  const float _function_support = 6.5f;
  unsigned int _fields_buffer_size;
  std::vector<float> _field_array;
};

class InterpolationShaderProg {
public:
  InterpolationShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::INTERP_FIELDS])
  {
  };

  void compute(std::shared_ptr<kp::ImageT<float>> fields, uint32_t width, uint32_t height);
  void record(
    std::shared_ptr<kp::Sequence> seq,
    std::shared_ptr<kp::ImageT<float>> fields,
    uint32_t width,
    uint32_t height);

  void update(
    uint32_t width,
    uint32_t height);

  float getSumQ() const { 
    auto sumQ = _tensors[ShaderBuffers::SUM_Q]->vector<float>()[0];
    return sumQ; 
  };
private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _interpAlgorithm;
  TensorMap& _tensors;
  //float _sum_Q = 0.0f;
};

class ForcesShaderProg {
public:
  ForcesShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FORCES])
  {
  }
  float compute(unsigned int num_points, float exaggeration);
  void record(
    std::shared_ptr<kp::Sequence> seq, 
    unsigned int num_points, 
    float exaggeration);

  void update(
    float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _forcesAlgorithm;
  TensorMap& _tensors;
};

class UpdateShaderProg {
public:
  UpdateShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::UPDATE])
  {
  }
  void compute(unsigned int num_points, float eta, float minimum_gain, float iteration, float momentumn, unsigned int momentum_switch, float final_momentum, float gain_mult);
  void record(
    std::shared_ptr<kp::Sequence> seq, 
    unsigned int num_points, 
    float eta, 
    float minimum_gain, 
    float iteration, 
    float momentum, 
    unsigned int momentum_switch, 
    float final_momentum, 
    float gain_mult);

  void update(
    float eta,
    float minimum_gain,
    float iteration,
    float momentum,
    unsigned int momentum_switch,
    float final_momentum,
    float gain_mult);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _updateAlgorithm;
  TensorMap& _tensors;
};

class CenterScaleShaderProg {
public:
  CenterScaleShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::CENTER_SCALE])
  {
  }
  std::vector<float> compute(unsigned int num_points, float exaggeration);

  void record(
    std::shared_ptr<kp::Sequence> seq, 
    unsigned int num_points, 
    float exaggeration);

  void update(
    float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _centerScaleAlgorithm;
  TensorMap& _tensors;
};
