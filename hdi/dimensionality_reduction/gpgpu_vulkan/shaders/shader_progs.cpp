#include "shader_progs.h"
#include <iostream> // for debug logging
#include <cstdlib> 
#include <cmath>
#include <algorithm>
#include "vulkan_extension.h"

std::vector<float> BoundsShaderProg::compute(float padding) {
  // Implement the compute function using _mgr and _tensors
  // This is a placeholder implementation
  //std::cout << "Bounds prog" << std::endl;
  std::vector<uint32_t> result;
  // Actual computation logic goes here
  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::NUM_POINTS]
  };
  auto algorithm = _mgr->algorithm(params, _shaderBinary, kp::Workgroup({ 1,1,1 }), {}, std::vector<float>({ padding }));
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(params)
    ->eval();

  return _tensors[ShaderBuffers::BOUNDS]->vector<float>();
}

/// <summary>
/// Record bounds calculation with padding.
/// 
/// This is the last step in the sequence and all relevant buffers should be
/// synchronized after this call.
/// </summary>
/// <param name="seq">The kp::Sequence being recordss</param>
/// <param name="padding">The padding to apply</param>
void BoundsShaderProg::record_padded(
  std::shared_ptr<kp::Sequence> seq,
  float padding) {
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::NUM_POINTS]
  };


  _boundsAlgorithmPadded = _mgr->algorithm(algoParams, _shaderBinary, kp::Workgroup({ 1,1,1 }), {}, std::vector<float>({ padding }));
    seq
      //->record<kp::OpSyncDevice>(std::vector<std::shared_ptr<kp::Memory>>({ _tensors[ShaderBuffers::BOUNDS] }))
      ->record<kp::OpAlgoDispatch>(_boundsAlgorithmPadded)
      ->record<kp::OpSyncLocal>(std::vector<std::shared_ptr<kp::Memory>>({ 
        _tensors[ShaderBuffers::BOUNDS],
        _tensors[ShaderBuffers::KLDIV],
        _tensors[ShaderBuffers::POSITION]
       }));
}

/// <summary>
/// Recalculate the bounds without padding.
/// </summary>
/// <param name="seq">The kp::Sequence being recorded</param>
void BoundsShaderProg::record_unpadded(
  std::shared_ptr<kp::Sequence> seq) {
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::NUM_POINTS]
  };

  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>({ _tensors[ShaderBuffers::BOUNDS] }),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);

  _boundsAlgorithmUnpadded = _mgr->algorithm(algoParams, _shaderBinary, kp::Workgroup({ 1,1,1 }), {}, std::vector<float>({ 0.0f }));
  seq
    //->record<kp::OpSyncDevice>(std::vector<std::shared_ptr<kp::Memory>>({ _tensors[ShaderBuffers::BOUNDS] }))
    ->record<kp::OpAlgoDispatch>(_boundsAlgorithmUnpadded)
    ->record(shaderBarrier);
}

std::shared_ptr<kp::ImageT<float>> StencilShaderProg::compute(
  uint32_t width, 
  uint32_t height, 
  unsigned int num_points, 
  std::vector<float> bounds) {
  //std::cout << "Stencil prog" << std::endl;
  auto npwidth = width;
  if (npwidth % 8) {
    npwidth += 8 - (npwidth % 8); // Align to next multiple of 8
  }

  auto stencil_array = std::vector<float>(npwidth * height * 4, 0.0); //npwidth 4
  auto stencil_out = _mgr->imageT<float>(stencil_array, npwidth, height, 4); //npwidth 4u
  auto pushConsts = std::vector<float>({ bounds[0], bounds[1], bounds[2], bounds[3], (float) width, (float) height });

  const std::vector<std::shared_ptr<kp::Memory>> params = { _tensors[ShaderBuffers::POSITION], stencil_out };
  auto algorithm = _mgr->algorithm(params, _shaderBinary, kp::Workgroup({ num_points, 1, 1 }), {}, pushConsts);
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(std::vector<std::shared_ptr<kp::Memory>>({ stencil_out }))
    ->eval();

  return stencil_out;
}


void StencilShaderProg::record(
  std::shared_ptr<kp::Sequence> seq,
  uint32_t width,
  uint32_t height,
  unsigned int num_points,
  std::vector<float> bounds,
  unsigned int new_fields_buffer_size) {

  _fields_buffer_size = new_fields_buffer_size;
  _stencil_array = std::vector<float>(_fields_buffer_size * _fields_buffer_size * 4, 0.0);
  _stencil_out = _mgr->imageT<float>(_stencil_array, _fields_buffer_size, _fields_buffer_size, 4);
  // Needs patch in Sequence.hpp
  auto cmdBuf = seq->getCommandBuffer();
  _stencil_out->recordPrimaryImageBarrier(
    *(cmdBuf.get()),
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eShaderWrite,
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::ImageLayout::eGeneral);

  const std::vector<std::shared_ptr<kp::Memory>> params = { 
    _tensors[ShaderBuffers::POSITION], _tensors[ShaderBuffers::UBO_STENCIL], _stencil_out};
  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>({ _stencil_out }),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);
  _stencilAlgorithm = _mgr->algorithm(params, _shaderBinary, kp::Workgroup({ num_points, 1, 1 }), {}, {});
  stencilParams uboVals = { {bounds[0], bounds[1]}, {bounds[2], bounds[3]}, {(float)width, (float)height} };
  _ubo.setData(uboVals, _stencilAlgorithm, 1);
  seq->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(_stencilAlgorithm)
    ->record(shaderBarrier);
}

// only valid if new_fields_buffer_size == _fields_buffer_size
void StencilShaderProg::update(
  uint32_t width,
  uint32_t height,
  std::vector<float> bounds,
  unsigned int new_fields_buffer_size) {
  assert(new_fields_buffer_size == _fields_buffer_size);
  std::fill(_stencil_array.begin(), _stencil_array.end(), 0.0f);
  _stencil_out->setData(_stencil_array);
  stencilParams uboVals = { {bounds[0], bounds[1]}, {bounds[2], bounds[3]}, {(float)width, (float)height} };
  _ubo.modifyData(uboVals, _stencilAlgorithm, 1);
}


std::shared_ptr<kp::ImageT<float>> FieldComputationShaderProg::compute(
  std::shared_ptr<kp::ImageT<float>> stencil, 
  uint32_t width, uint32_t height) {
  //std::cout << "Field comp prog" << std::endl;
  auto npwidth = width;
  if (npwidth % 8) {
    npwidth += 8 - (npwidth % 8); // Align to next multiple of 8
  }

  auto field_out = _mgr->image(std::vector<float>(npwidth * height * 4, 0.0f), npwidth, height, 4); // npwidth
  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::BOUNDS],
      field_out,
      stencil,
      _tensors[ShaderBuffers::NUM_POINTS]
  };
  auto pushConsts = std::vector<float>({ (float) width, (float) height, _function_support });
  auto algorithm = _mgr->algorithm(params, _shaderBinary, kp::Workgroup({ width, height, 1 }), {}, pushConsts);

  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(std::vector<std::shared_ptr<kp::Memory>>({ field_out }))
    ->eval();

  return field_out;
}

void FieldComputationShaderProg::record(
  std::shared_ptr<kp::Sequence> seq, 
  std::shared_ptr<kp::ImageT<float>> stencil,
  uint32_t width, 
  uint32_t height,
  unsigned int new_fields_buffer_size) {
  _fields_buffer_size = new_fields_buffer_size;
  _field_array = std::vector<float>(_fields_buffer_size * _fields_buffer_size * 4, 0.0f);
  _field_out = _mgr->image(_field_array, _fields_buffer_size, _fields_buffer_size, 4); // npwidth
  // requires patch in Sequence.hpp
  auto cmdBuf = seq->getCommandBuffer();
  _field_out->recordPrimaryImageBarrier(
    *(cmdBuf.get()),
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eShaderWrite,
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::ImageLayout::eGeneral);
  /*_field_out->recordPrimaryImageBarrier(
    *(cmdBuf.get()),
    vk::AccessFlagBits::eTransferRead,
    vk::AccessFlagBits::eShaderWrite,
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::ImageLayout::eGeneral);*/

  //stencil->recordPrimaryImageBarrier(
  //  *(cmdBuf.get()),
  //  vk::AccessFlagBits::eTransferWrite,
  //  vk::AccessFlagBits::eShaderRead,
  //  vk::PipelineStageFlagBits::eTransfer,
  //  vk::PipelineStageFlagBits::eComputeShader,
  //  vk::ImageLayout::eGeneral);
  auto& dispatchTensor = _tensors[ShaderBuffers::IMAGE_WORKGROUP];
  uint32_t dispatchData[3] = { width, height, 1 };
  dispatchTensor->setData((void*)dispatchData, 3 * sizeof(uint32_t));
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::BOUNDS],
      _field_out,
      stencil,
      _tensors[ShaderBuffers::NUM_POINTS],
      _tensors[ShaderBuffers::UBO_FIELD]
  };
  _fieldAlgorithm = _mgr->algorithm(algoParams, _shaderBinary, {}, {}, {});
  const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
      _tensors[ShaderBuffers::BOUNDS],
      _field_out,
      _tensors[ShaderBuffers::NUM_POINTS],
      _tensors[ShaderBuffers::UBO_FIELD]
  };
  fieldParams uboVals = { {(float)width, (float)height}, _function_support };
  _ubo.setData(uboVals, _fieldAlgorithm, 5);
  const vk::AccessFlags readWriteFlags = vk::AccessFlags(vk::AccessFlagBits::eShaderWrite) | vk::AccessFlags(vk::AccessFlagBits::eShaderRead);
  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>({ _field_out }),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);

  seq->record<kp::OpSyncDevice>({ dispatchTensor })
    //->record(dispatchBarrier)
    ->record<kp::OpSyncDevice>(syncParams)
    ->record<OpIndirectDispatch>(_fieldAlgorithm, dispatchTensor)
    ->record(shaderBarrier);
}

void FieldComputationShaderProg::update(
  uint32_t width,
  uint32_t height,
  unsigned int new_fields_buffer_size) {
  assert(new_fields_buffer_size == _fields_buffer_size);
  
  auto& dispatchTensor = _tensors[ShaderBuffers::IMAGE_WORKGROUP];
  uint32_t dispatchData[3] = { width, height, 1 };
  dispatchTensor->setData((void*)dispatchData, 3 * sizeof(uint32_t));
  std::fill(_field_array.begin(), _field_array.end(), 0.0f);
  _field_out->setData(_field_array);
  fieldParams uboVals = { {(float)width, (float)height}, _function_support };
  _ubo.modifyData(uboVals, _fieldAlgorithm, 5);
  
}

void InterpolationShaderProg::compute(
  std::shared_ptr<kp::ImageT<float>> fields, 
  uint32_t width, uint32_t height) {

  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::INTERP_FIELDS],
      _tensors[ShaderBuffers::SUM_Q],
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::NUM_POINTS],
      fields
  };

  auto pushConsts = std::vector<float>({ (float) width, (float) height });
  auto algorithm = _mgr->algorithm(params, _shaderBinary, kp::Workgroup({ 1, 1, 1 }), {}, pushConsts);
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(params)
    ->eval();
}

void InterpolationShaderProg::record(
  std::shared_ptr<kp::Sequence> seq,
  std::shared_ptr<kp::ImageT<float>> fields,
  uint32_t width,
  uint32_t height) {
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::INTERP_FIELDS],
      _tensors[ShaderBuffers::SUM_Q],
      fields,
      _tensors[ShaderBuffers::NUM_POINTS],
      _tensors[ShaderBuffers::UBO_INTERP]
  };
  _interpAlgorithm = _mgr->algorithm(algoParams, _shaderBinary, kp::Workgroup({ 1, 1, 1 }), {}, {});
  interpParams uboVals = { {(float)width, (float)height} };
  _ubo.setData(uboVals, _interpAlgorithm, 6);
  const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
      _tensors[ShaderBuffers::INTERP_FIELDS],
      _tensors[ShaderBuffers::SUM_Q],
      _tensors[ShaderBuffers::UBO_INTERP]
  };

  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>(
      { _tensors[ShaderBuffers::INTERP_FIELDS], 
        _tensors[ShaderBuffers::SUM_Q]}),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpSyncDevice>(syncParams)
    ->record<kp::OpAlgoDispatch>(_interpAlgorithm)
    ->record(shaderBarrier);
}

void InterpolationShaderProg::update(
  uint32_t width,
  uint32_t height) {
  interpParams uboVals = { {(float)width, (float)height} };
  _ubo.modifyData(uboVals, _interpAlgorithm, 6);
}

float ForcesShaderProg::compute(unsigned int num_points, float exaggeration) {
  //std::cout << "Forces comp prog" << std::endl;
  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::NEIGHBOUR],
      _tensors[ShaderBuffers::INDEX],
      _tensors[ShaderBuffers::PROBABILITIES],
      _tensors[ShaderBuffers::INTERP_FIELDS],
      _tensors[ShaderBuffers::GRADIENTS],
      _tensors[ShaderBuffers::NUM_POINTS],
      _tensors[ShaderBuffers::SUM_Q],
      _tensors[ShaderBuffers::KLDIV]
  };

  auto pushConsts = std::vector<float>({ exaggeration});
  auto grid_size = unsigned int(sqrt(num_points) + 1);
  auto algorithm = _mgr->algorithm(
    params, 
    _shaderBinary, 
    kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(params)
    ->eval();
  return _tensors[ShaderBuffers::KLDIV]->data<float>()[0];
}

void ForcesShaderProg::record(
  std::shared_ptr<kp::Sequence> seq,
  unsigned int num_points,
  float exaggeration) {
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
    _tensors[ShaderBuffers::POSITION],
    _tensors[ShaderBuffers::NEIGHBOUR],
    _tensors[ShaderBuffers::INDEX],
    _tensors[ShaderBuffers::PROBABILITIES],
    _tensors[ShaderBuffers::INTERP_FIELDS],
    _tensors[ShaderBuffers::GRADIENTS],
    _tensors[ShaderBuffers::NUM_POINTS],
    _tensors[ShaderBuffers::SUM_Q],
    _tensors[ShaderBuffers::KLDIV],
    _tensors[ShaderBuffers::UBO_FORCES]
  };
  auto grid_size = unsigned int(sqrt(num_points) + 1);
  _forcesAlgorithm = _mgr->algorithm(algoParams, _shaderBinary, kp::Workgroup({ grid_size, grid_size, 1 }), {}, {});
  forcesParams uboVals = { {exaggeration} };
  _ubo.setData(uboVals, _forcesAlgorithm, 9);
  const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
    _tensors[ShaderBuffers::NEIGHBOUR],
    _tensors[ShaderBuffers::INDEX],
    _tensors[ShaderBuffers::PROBABILITIES],
    _tensors[ShaderBuffers::GRADIENTS],
    _tensors[ShaderBuffers::KLDIV],
    _tensors[ShaderBuffers::UBO_FORCES]
  };
  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>(
      { _tensors[ShaderBuffers::GRADIENTS],
        _tensors[ShaderBuffers::KLDIV] }),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);

  seq->record<kp::OpSyncDevice>(syncParams)
    ->record<kp::OpAlgoDispatch>(_forcesAlgorithm)
    ->record(shaderBarrier);
}

void ForcesShaderProg::update(
  float exaggeration) {
  forcesParams uboVals = { {exaggeration} };
  _ubo.modifyData(uboVals, _forcesAlgorithm, 9);
}

void UpdateShaderProg::compute(unsigned int num_points, float eta, float minimum_gain, float iteration, float momentum, unsigned int momentum_switch, float final_momentum, float gain_mult) {
  //std::cout << "Update prog" << std::endl;
  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::GRADIENTS],
      _tensors[ShaderBuffers::PREV_GRADIENTS],
      _tensors[ShaderBuffers::GAIN],
      _tensors[ShaderBuffers::NUM_POINTS]
  };

  auto num_workgroups = unsigned int((num_points * 2 / 64) + 1);
  auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
  auto pushConsts = std::vector<float>({ 
    eta, minimum_gain, 
    float(iteration), float(momentum_switch), 
    momentum, final_momentum, gain_mult });
  auto algorithm = _mgr->algorithm(
    params, 
    _shaderBinary, 
    kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(params)
    ->eval();
}

void UpdateShaderProg::record(
  std::shared_ptr<kp::Sequence> seq,
  unsigned int num_points,
  float eta,
  float minimum_gain,
  float iteration,
  float momentum,
  unsigned int momentum_switch,
  float final_momentum,
  float gain_mult) {
    const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
        _tensors[ShaderBuffers::POSITION],
        _tensors[ShaderBuffers::GRADIENTS],
        _tensors[ShaderBuffers::PREV_GRADIENTS],
        _tensors[ShaderBuffers::GAIN],
        _tensors[ShaderBuffers::NUM_POINTS],
        _tensors[ShaderBuffers::UBO_UPDATE]
    };
    auto num_workgroups = unsigned int((num_points * 2 / 64) + 1);
    auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
    _updateAlgorithm = _mgr->algorithm(
      algoParams, 
      _shaderBinary, 
      kp::Workgroup({ grid_size, grid_size, 1 }), {}, {});
    updaterParams uboVals = { 
      eta, minimum_gain, 
      iteration, momentum_switch, 
      momentum, final_momentum, 
      gain_mult };
    _ubo.setData(uboVals, _updateAlgorithm, 5);
    const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
        _tensors[ShaderBuffers::UBO_UPDATE]
    };
    auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
      std::vector<std::shared_ptr<kp::Memory>>(
        { _tensors[ShaderBuffers::PREV_GRADIENTS],
          _tensors[ShaderBuffers::GAIN],
          _tensors[ShaderBuffers::POSITION] }),
      vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader);
    seq->record<kp::OpSyncDevice>(syncParams)
      ->record<kp::OpAlgoDispatch>(_updateAlgorithm)
      ->record(shaderBarrier);
  }

void UpdateShaderProg::update(
  float eta,
  float minimum_gain,
  float iteration,
  float momentum,
  unsigned int momentum_switch,
  float final_momentum,
  float gain_mult) {
  updaterParams uboVals = {
    eta, minimum_gain,
    iteration, momentum_switch,
    momentum, final_momentum,
    gain_mult };
  _ubo.modifyData(uboVals, _updateAlgorithm, 5);
}

std::vector<float> CenterScaleShaderProg::compute(unsigned int num_points, float exaggeration) {
  //std::cout << "Center-Scale prog" << std::endl;
  const std::vector<std::shared_ptr<kp::Memory>> params = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::NUM_POINTS]
  };
  auto scale = 0.0f;
  auto diameter = 0.0f;

  if (exaggeration > 1.2) {
    scale = 1.0f;
    diameter = 0.1;
  }

  auto pushConsts = std::vector<float>({ scale, diameter });
  auto num_workgroups = unsigned int(num_points / 128) + 1;
  auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
  auto algorithm = _mgr->algorithm(
    params, 
    _shaderBinary, 
    kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
  auto seq = _mgr->sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algorithm)
    ->record<kp::OpSyncLocal>(params)
    ->eval();

  return _tensors[ShaderBuffers::POSITION]->vector<float>();
}

void CenterScaleShaderProg::record(
  std::shared_ptr<kp::Sequence> seq,
  unsigned int num_points,
  float exaggeration) {
  const std::vector<std::shared_ptr<kp::Memory>> algoParams = {
      _tensors[ShaderBuffers::POSITION],
      _tensors[ShaderBuffers::BOUNDS],
      _tensors[ShaderBuffers::NUM_POINTS],
      _tensors[ShaderBuffers::UBO_CENTER_SCALE]
  };
  auto scale = 0.0f;
  auto diameter = 0.0f;

  if (exaggeration > 1.2) {
    scale = 1.0f;
    diameter = 0.1;
  }

  auto num_workgroups = unsigned int(num_points / 128) + 1;
  auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
  _centerScaleAlgorithm = _mgr->algorithm(
    algoParams,
    _shaderBinary, 
    kp::Workgroup({ grid_size, grid_size, 1 }), {}, {});
  centerScaleParams uboVals = { scale, diameter};
  _ubo.setData(uboVals, _centerScaleAlgorithm, 3);
  const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
      _tensors[ShaderBuffers::UBO_CENTER_SCALE]
  };
  auto shaderBarrier = std::make_shared<kp::OpMemoryBarrier>(
    std::vector<std::shared_ptr<kp::Memory>>(
      { _tensors[ShaderBuffers::POSITION] }),
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpSyncDevice>(syncParams)
    ->record<kp::OpAlgoDispatch>(_centerScaleAlgorithm)
    ->record(shaderBarrier);
}

void CenterScaleShaderProg::update(
  float exaggeration) {
  auto scale = 0.0f;
  auto diameter = 0.0f;

  if (exaggeration > 1.2) {
    scale = 1.0f;
    diameter = 0.1;
  }
  centerScaleParams uboVals = { scale, diameter };
  _ubo.modifyData(uboVals, _centerScaleAlgorithm, 3);

}

