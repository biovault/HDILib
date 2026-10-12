#pragma once

#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/shaders.h"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/vkUniformBufferHelper.h"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/tensor_config.h"

#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>

class ShaderImageHelper {
  public:
  ShaderImageHelper() {};

  void createBuffers(
    std::shared_ptr<kp::Manager> mgr, 
    uint32_t fields_buffer_size,
    uint32_t num_pnts,
    uint32_t threads_per_workgroup_fields = 256) {
    _stencil_array = std::vector<float>(fields_buffer_size * fields_buffer_size * 4, 0.0);
    _stencil_out = mgr->imageT<float>(_stencil_array, fields_buffer_size, fields_buffer_size, 4, vk::ImageTiling::eOptimal);
    _field_array = std::vector<float>(fields_buffer_size * fields_buffer_size * 4, 0.0f);
    _field_out = mgr->imageT<float>(_field_array, fields_buffer_size, fields_buffer_size, 4, vk::ImageTiling::eOptimal);
    _field_sample = _field_out->createSampledView();
    _activePixelList = mgr->tensorT<uint32_t>(std::vector<uint32_t>(fields_buffer_size * fields_buffer_size * 2, 0));
    // Create partial results buffer - don't know num active points so cover all field
    _num_workgroups = (num_pnts + threads_per_workgroup_fields - 1) / threads_per_workgroup_fields;
    uint32_t partialSize = _num_workgroups * fields_buffer_size * fields_buffer_size * 4;
    _partialResults = mgr->tensorT(std::vector<float>(partialSize, 0.0f));
  };
  std::weak_ptr<kp::ImageT<float>> getStencilImage() const {
    return _stencil_out;
  };
  std::vector<float>& getStencilArray() {
    return _stencil_array;
  };
  std::weak_ptr<kp::ImageT<float>> getFieldImage() const {
    return _field_out;
  };
  std::weak_ptr<kp::ImageT<float>> getFieldSamplerImage() const {
    return _field_sample;
  };
  std::vector<float>& getFieldArray() {
    return _field_array;
  };
  std::weak_ptr<kp::TensorT<uint32_t>> getActivePixelList() const {
    return _activePixelList;
  };
  std::weak_ptr<kp::TensorT<float>> getPartialResults() const {
    return _partialResults;
  };

  void setFieldArraySampler(vk::Sampler sampler) {
    _field_sample->setSampler(sampler);
  };

  uint32_t getNumWorkgroups() const {
    return _num_workgroups;
  };

  void clearBuffers() {
    #pragma omp for
    for (int i = 0; i < static_cast<int>(_stencil_array.size()); ++i) {
      _stencil_array[i] = 0.0f;
    }
    _stencil_out->setData(_stencil_array);
    #pragma omp for
    for (int i = 0; i < static_cast<int>(_field_array.size()); ++i) {
      _field_array[i] = 0.0f;
    }
    _field_out->setData(_field_array);
  };
  
  void resetBuffers() {
    _stencil_out->destroy();
    _field_out->destroy();
    _field_sample->destroy();
    _activePixelList->destroy();
    _partialResults->destroy();
    _stencil_out.reset();
    _field_out.reset();
    _field_sample.reset();
    _activePixelList.reset();
    _partialResults.reset();
  }
  
  

private:
  std::shared_ptr<kp::ImageT<float>> _stencil_out;
  std::vector<float> _stencil_array;
  std::shared_ptr<kp::ImageT<float>> _field_out;
  std::vector<float> _field_array;
  std::shared_ptr<kp::ImageT<float>> _field_sample;
  std::shared_ptr<kp::TensorT<uint32_t>> _activePixelList;
  std::shared_ptr<kp::TensorT<float>> _partialResults;
  uint32_t _num_workgroups;
};

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

  void record_padded(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points,
    float padding = 0.1);

  void record_unpadded(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points);

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
    _ubo(mgr, sizeof(stencilParams))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t width, uint32_t height,
    std::weak_ptr<kp::ImageT<float>> stencil,
    unsigned int num_points,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t width, uint32_t height,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);
  
  void clear() {
    _stencilAlgorithm->destroy();
    _stencilAlgorithm.reset();
  }

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _stencilAlgorithm;
  TensorMap& _tensors;
  unsigned int _fields_buffer_size;
  UniformBufferHelper _ubo;

};

class Stencil2ListShaderProg {
public:
  Stencil2ListShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinaryClearCounter(getSPIRVBinaries()[SPIRVShader::RESET_COUNTER]),
    _shaderBinary2List(getSPIRVBinaries()[SPIRVShader::STENCIL2ACTIVE]),
    _fields_buffer_size(0),
    _ubo(mgr, sizeof(stencilParams))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t width, uint32_t height,
    std::weak_ptr<kp::ImageT<float>> stencil,
    std::weak_ptr<kp::TensorT<uint32_t>> activePixelList,
    unsigned int num_points,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t width, uint32_t height,
    std::vector<float> bounds,
    unsigned int new_fields_buffer_size = 0);
  
  void clear() {
    _clearCounterAlgorithm->destroy();
    _stencil2listAlgorithm->destroy();
    _fieldWorkgroupAlgorithm->destroy();
  }

private:
  std::vector<uint32_t>& _shaderBinaryClearCounter;
  std::vector<uint32_t>& _shaderBinary2List;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _clearCounterAlgorithm;
  std::shared_ptr<kp::Algorithm> _stencil2listAlgorithm;
  std::shared_ptr<kp::Algorithm> _fieldWorkgroupAlgorithm;
  TensorMap& _tensors;
  unsigned int _fields_buffer_size;
  UniformBufferHelper _ubo;

};

class FieldComputationShaderProg {
public:
  FieldComputationShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FIELDS]),
    _fields_buffer_size(0),
    _ubo(mgr, sizeof(fieldParams))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points,
    uint32_t width, uint32_t height,
    std::shared_ptr<kp::ImageT<float>> sampleFields,
    std::shared_ptr<kp::ImageT<float>> fields,
    std::shared_ptr<kp::ImageT<float>> stencil,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t num_points,
    uint32_t width, uint32_t height,
    unsigned int new_fields_buffer_size = 0);
  
  void clear() {
    _fieldAlgorithm->destroy();
  }
  
private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _fieldAlgorithm;
  TensorMap& _tensors;
  const float _function_support = 6.5f;
  unsigned int _fields_buffer_size;
  UniformBufferHelper _ubo;
};

class FieldComputationEnhShaderProg {
public:
  FieldComputationEnhShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FIELDS_ENH]),
    _shaderFieldWorkgroup(getSPIRVBinaries()[SPIRVShader::FIELD_WORKGROUP]),
    _fields_buffer_size(0),
    _ubo(mgr, sizeof(fieldParams))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points,
    uint32_t width, uint32_t height,
    std::weak_ptr<kp::ImageT<float>> sampleFields,
    std::weak_ptr<kp::ImageT<float>> fields,
    std::weak_ptr<kp::TensorT<uint32_t>> activePixelList,
    unsigned int new_fields_buffer_size = 0);

  void update(
    uint32_t num_points,
    uint32_t width, uint32_t height,
    unsigned int new_fields_buffer_size = 0);
  
  void clear() {
    _fieldAlgorithm->destroy();
  }
  
private:
  std::vector<uint32_t>& _shaderBinary;
  std::vector<uint32_t>& _shaderFieldWorkgroup;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _fieldWorkgroupAlgorithm;
  std::shared_ptr<kp::Algorithm> _fieldAlgorithm;
  TensorMap& _tensors;
  const float _function_support = 6.5f;
  unsigned int _fields_buffer_size;
  UniformBufferHelper _ubo;
};

class InterpolationShaderProg {
public:
  InterpolationShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::INTERP_FIELDS]),
    _ubo(mgr, sizeof(interpParams))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points,
    std::weak_ptr<kp::ImageT<float>> sampleFields,
    std::weak_ptr<kp::ImageT<float>> fields,
    uint32_t width,
    uint32_t height);

  void update(
    uint32_t num_points,
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
  UniformBufferHelper _ubo;
  //float _sum_Q = 0.0f;
};

class InterpolationEnhShaderProg {
public:
  InterpolationEnhShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary1(getSPIRVBinaries()[SPIRVShader::INTERP_FIELDS_ENH1]),
    _shaderBinary2(getSPIRVBinaries()[SPIRVShader::INTERP_FIELDS_ENH2]),
    _ubo(mgr, sizeof(interpParams)),
    _ubo2(mgr, sizeof(interp2Params))
  {
  };

  void record(
    std::shared_ptr<kp::Sequence> seq,
    uint32_t num_points,
    uint32_t num_workgroups,
    std::weak_ptr<kp::ImageT<float>> sampleFields,
    std::weak_ptr<kp::ImageT<float>> fields,
    uint32_t width,
    uint32_t height);

  void update(
    uint32_t num_points,
    uint32_t width,
    uint32_t height);

  float getSumQ() const {
    auto sumQ = _tensors[ShaderBuffers::SUM_Q]->vector<float>()[0];
    return sumQ;
  };
  
  void clear() {
    _interpAlgorithm1->destroy();
    _interpAlgorithm2->destroy();
  }
private:
  std::vector<uint32_t>& _shaderBinary1;
  std::vector<uint32_t>& _shaderBinary2;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _interpAlgorithm1;
  std::shared_ptr<kp::Algorithm> _interpAlgorithm2;
  TensorMap& _tensors;
  UniformBufferHelper _ubo;
  UniformBufferHelper _ubo2;
  //float _sum_Q = 0.0f;
};

class ForcesShaderProg {
public:
  ForcesShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::COMPUTE_FORCES]),
    _ubo(mgr, sizeof(forcesParams))
  {
  }

  void record(
    std::shared_ptr<kp::Sequence> seq, 
    unsigned int num_points, 
    float exaggeration);

  void update(
    uint32_t num_points,
    float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _forcesAlgorithm;
  TensorMap& _tensors;
  UniformBufferHelper _ubo;
};

class UpdateShaderProg {
public:
  UpdateShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::UPDATE]),
    _ubo(mgr, sizeof(updaterParams))
  {
  }

  void record(
    std::shared_ptr<kp::Sequence> seq, 
    uint32_t num_points, 
    float eta, 
    float minimum_gain, 
    float iteration, 
    float momentum, 
    unsigned int momentum_switch, 
    float final_momentum, 
    float gain_mult);

  void update(
    uint32_t num_points,
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
  UniformBufferHelper _ubo;
};

class CenterScaleShaderProg {
public:
  CenterScaleShaderProg(std::shared_ptr<kp::Manager> mgr, TensorMap& tensors) :
    _mgr(mgr),
    _tensors(tensors),
    _shaderBinary(getSPIRVBinaries()[SPIRVShader::CENTER_SCALE]),
    _ubo(mgr, sizeof(centerScaleParams))
  {
  }

  void record(
    std::shared_ptr<kp::Sequence> seq, 
    uint32_t num_points,
    float exaggeration);

  void update(
    uint32_t num_points,
    float exaggeration);

private:
  std::vector<uint32_t>& _shaderBinary;
  std::shared_ptr<kp::Manager> _mgr;
  std::shared_ptr<kp::Algorithm> _centerScaleAlgorithm;
  TensorMap& _tensors;
  UniformBufferHelper _ubo;
};
