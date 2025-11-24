#include "gpgpu_sne_comp_vulkan.h"

#include <vector>
#include <limits> 
#include <iostream>
#include <cmath> // for sqrt
#include <memory>
#include <sstream>
#include <chrono>
#include "tensor_config.h"
#include "shaders/shaders.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace hdi {
  namespace dr {

    // Linearized sparse neighbourhood matrix
    struct LinearProbabilityMatrix {
      std::vector<uint32_t> neighbours;
      std::vector<float> probabilities;
      std::vector<int> indices;
    };

    GpgpuSneVulkan::GpgpuSneVulkan() :
      _initialized(false),
      _adaptive_resolution(true),
      _resolutionScaling(RESOLUTION_SCALING),
      kl_divergence(-1.0f) // Initialize KL divergence to -1.0f
    {

    }

    std::vector<float> GpgpuSneVulkan::computeInitialBounds(const embedding_type* embedding, float padding) {
      const float* const points = embedding->getContainer().data();

      std::vector<float> bounds({
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
      });

      for (int i = 0; i < embedding->numDataPoints(); ++i) {
        float x = points[i * 2 + 0];
        float y = points[i * 2 + 1];

        bounds[0] = std::min<float>(x, bounds[0]);
        bounds[2] = std::max<float>(x, bounds[2]);
        bounds[1] = std::min<float>(y, bounds[1]);
        bounds[3] = std::max<float>(y, bounds[3]);
      }

      // Add any extra padding if requested
      if (padding != 0) {
        float half_padding = padding / 2.0;

        float x_padding = (bounds[2] - bounds[0]) * half_padding;
        float y_padding = (bounds[3] - bounds[1]) * half_padding;

        bounds[0] -= x_padding;
        bounds[2] += x_padding;
        bounds[1] -= y_padding;
        bounds[3] += y_padding;
      }

      return bounds;
    }
    void GpgpuSneVulkan::initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P) {
      _params = params;

      unsigned int num_points = embedding->numDataPoints();
      /*if (num_points < 10000) {
        _resolutionScaling = 4;
      }
      else if (num_points < 1000) {
        _resolutionScaling = 4;
      }*/
      if (num_points < 1000) {
        _resolutionScaling = 4;
      }

      // Linearize sparse probability matrix
      LinearProbabilityMatrix linear_P;
      unsigned int num_pnts = embedding->numDataPoints();
      for (int i = 0; i < num_pnts; ++i) {
        linear_P.indices.push_back(linear_P.neighbours.size());
        int size = 0;
        for (const auto& pij : P[i]) {
          linear_P.neighbours.push_back(pij.first);
          linear_P.probabilities.push_back(pij.second);
          size++;
        }
        linear_P.indices.push_back(size);
      }

      // Compute initial data bounds
      _bounds = computeInitialBounds(embedding, 0.1f);

      // Initialize all Vulkan resources
      initializeVulkan(num_points, linear_P);

      _initialized = true;
    }

    void GpgpuSneVulkan::initializeVulkan(unsigned int num_pnts, const LinearProbabilityMatrix& linear_P) {
      // Create the manager with debug extensions
      _mgr = std::make_shared<kp::Manager>(0, std::vector<uint32_t>(), std::vector<std::string>({ "VK_KHR_synchronization2" }));
      // <DEBUG output physical device info>
      vk::PhysicalDeviceSynchronization2FeaturesKHR sync2Features{};
      vk::PhysicalDeviceFeatures2 features2{};
      features2.pNext = &sync2Features;

      _mgr->getPhysicalDevice()->getFeatures2(&features2);
      std::cout << "Synchronization2 supported: "
        << (sync2Features.synchronization2 ? "YES" : "NO") << std::endl;

      auto pprops = _mgr->getPhysicalDevice()->getProperties();
      printf("Physical device API version: %u.%u.%u\n",
        VK_VERSION_MAJOR(pprops.apiVersion),
        VK_VERSION_MINOR(pprops.apiVersion),
        VK_VERSION_PATCH(pprops.apiVersion));

      vk::PhysicalDeviceSynchronization2FeaturesKHR sync2{};

      vk::PhysicalDeviceFeatures2 qfeatures2;
      qfeatures2.pNext = &sync2;
      _mgr->getPhysicalDevice()->getFeatures2(&qfeatures2);
      printf("sync2 feature supported/enabled = %u\n", sync2.synchronization2);
      auto p_khr = (void*)vkGetDeviceProcAddr(*_mgr->getDevice().get(), "vkCmdPipelineBarrier2KHR");
      auto p_core = (void*)vkGetDeviceProcAddr(*_mgr->getDevice().get(), "vkCmdPipelineBarrier2");
      printf("vkCmdPipelineBarrier2KHR = %p\n", p_khr);
      printf("vkCmdPipelineBarrier2     = %p\n", p_core);

      auto props = _mgr->getPhysicalDevice()->enumerateDeviceExtensionProperties();
      for (auto& p : props)
        std::cout << p.extensionName << std::endl;
      // </DEBUG output physical device info>

      _tensors[ShaderBuffers::POSITION] = _mgr->tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::INTERP_FIELDS] = _mgr->tensorT(std::vector<float>(num_pnts * 4, 0.0f));
      _tensors[ShaderBuffers::SUM_Q] = _mgr->tensorT(std::vector<float>(1, 0.0f));
      _tensors[ShaderBuffers::KLDIV] = _mgr->tensorT(std::vector<float>(1, 0.0f));
      _tensors[ShaderBuffers::GRADIENTS] = _mgr->tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::NEIGHBOUR] = _mgr->tensorT(linear_P.neighbours);
      _tensors[ShaderBuffers::PROBABILITIES] = _mgr->tensorT(linear_P.probabilities);
      _tensors[ShaderBuffers::INDEX] = _mgr->tensorT(linear_P.indices);
      _tensors[ShaderBuffers::PREV_GRADIENTS] = _mgr->tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::GAIN] = _mgr->tensorT(std::vector<float>(num_pnts * 2, 1.0f));
      _tensors[ShaderBuffers::BOUNDS] = _mgr->tensorT(std::vector<float>(4, 1.0f));
      _tensors[ShaderBuffers::NUM_POINTS] = _mgr->tensorT(std::vector<unsigned int>(1, num_pnts));
      _tensors[ShaderBuffers::IMAGE_WORKGROUP] = _mgr->tensorT(std::vector<unsigned int>(3, 1u));
      _tensors[ShaderBuffers::DEBUG] = _mgr->tensorT(std::vector<float>(num_pnts * 4, 0.0f));
      _numInterpWorkgroups = uint32_t((num_pnts + 127) / 128);
      std::cout << "Interoplat using " << _numInterpWorkgroups << " workgroups\n";
      _tensors[ShaderBuffers::PARTIAL_SUM] = _mgr->tensor(std::vector<float>(_numInterpWorkgroups, 0.0f));
      _tensors[ShaderBuffers::ATOMIC_COUNTER] = _mgr->tensorT(std::vector<uint32_t>(1, 0u));
      // These tensors will be used as UBOs
      _tensors[ShaderBuffers::UBO_STENCIL] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(stencilParams), 0));
      _tensors[ShaderBuffers::UBO_FIELD] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(fieldParams), 0));
      _tensors[ShaderBuffers::UBO_INTERP] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(interpParams), 0));
      _tensors[ShaderBuffers::UBO_INTERP2] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(interp2Params), 0));
      _tensors[ShaderBuffers::UBO_FORCES] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(forcesParams), 0));
      _tensors[ShaderBuffers::UBO_UPDATE] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(updaterParams), 0));
      _tensors[ShaderBuffers::UBO_CENTER_SCALE] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(centerScaleParams), 0));


      _boundsProg = std::make_shared<BoundsShaderProg>(_mgr, _tensors);
      _stencilProg = std::make_shared<StencilShaderProg>(_mgr, _tensors);
      _stencil2ListProg = std::make_shared<Stencil2ListShaderProg>(_mgr, _tensors);
      _fieldCompProg = std::make_shared<FieldComputationShaderProg>(_mgr, _tensors);
      _fieldCompEnhProg = std::make_shared<FieldComputationEnhShaderProg>(_mgr, _tensors);
      _interpProg = std::make_shared<InterpolationShaderProg>(_mgr, _tensors);
      _interpEnhProg = std::make_shared<InterpolationEnhShaderProg>(_mgr, _tensors);
      _forcesProg = std::make_shared<ForcesShaderProg>(_mgr, _tensors);
      _updateProg = std::make_shared<UpdateShaderProg>(_mgr, _tensors);
      _centerScaleProg = std::make_shared<CenterScaleShaderProg>(_mgr, _tensors);
    }

    void GpgpuSneVulkan::clean() {
      for (auto& n : _tensors)
        n.second->destroy();
      _tensors.clear();
      _initialized = false;
    }

    void GpgpuSneVulkan::compute(embedding_type* embedding, float exaggeration, float iteration, float mult) {
#ifdef SHADER_USE_PUSH_CONSTANTS
      compute_stepwise(embedding, exaggeration, iteration, mult);
#else
      compute_sequence(embedding, exaggeration, iteration, mult);
#endif // SHADER_USE_PUSH_CONSTANTS

    }

    void GpgpuSneVulkan::record_compute_sequence(
      float iteration,
      uint32_t width,
      uint32_t height,
      uint32_t num_points,
      float* bounds,
      float exaggeration,
      float mult) {

      _seq0 = _mgr->sequence();
      _seq0->begin();
      _shaderImageHelper.createBuffers(_mgr, _fields_buffer_size);
      _shaderImageHelper.setFieldArraySampler(_mgr->createLinearSampler());
      _stencilProg->record(_seq0, width, height, _shaderImageHelper.getStencilImage(), num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _stencil2ListProg->record(_seq0, _fields_buffer_size, _fields_buffer_size, _shaderImageHelper.getStencilImage(), _shaderImageHelper.getActivePixelList(), num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _fieldCompEnhProg->record(_seq0, num_points, width, height, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), _shaderImageHelper.getActivePixelList(), _fields_buffer_size);
      _interpEnhProg->record(_seq0, num_points, _numInterpWorkgroups, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), width, height);
      _seq0->end();
      /*_seq1 = _mgr->sequence();
      _seq1->begin();
      _fieldCompEnhProg->record(_seq1, num_points, width, height, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), _shaderImageHelper.getActivePixelList(), _fields_buffer_size);
      //_fieldCompProg->record(_seq0, num_points, width, height, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), _shaderImageHelper.getStencilImage(), _fields_buffer_size);
      //_interpProg->record(_seq0, num_points, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), width, height);
      _seq1->end();
      _seq2 = _mgr->sequence();
      _seq2->begin();
      _interpEnhProg->record(_seq2, num_points, _numInterpWorkgroups, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), width, height);
      _seq2->end();*/
      // Last sequence has constant size buffers is recorded once sequence for forces and update
      if (_seq3.get() == nullptr) {
        _seq3 = _mgr->sequence();
        _seq3->begin();
        _forcesProg->record(_seq3, num_points, exaggeration);
        _updateProg->record(_seq3, num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
        _boundsProg->record_unpadded(_seq3, num_points);
        _centerScaleProg->record(_seq3, num_points, exaggeration);
        _boundsProg->record_padded(_seq3, num_points, 0.1f);
        _seq3->end();
      }
    }

    void GpgpuSneVulkan::update_compute_sequence(
      float iteration,
      uint32_t num_points,
      uint32_t width, 
      uint32_t height, 
      float* bounds, 
      float exaggeration, 
      float mult) {
      _shaderImageHelper.clearBuffers();
      _stencilProg->update(width, height, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _stencil2ListProg->update(_fields_buffer_size, _fields_buffer_size, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      //_fieldCompProg->update(num_points, width, height, _fields_buffer_size);
      _fieldCompEnhProg->update(num_points, width, height, _fields_buffer_size);
      //_interpProg->update(num_points, width, height);
      _interpEnhProg->update(num_points, width, height);
      _forcesProg->update(num_points, exaggeration);
      _updateProg->update(num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
      _centerScaleProg->update(num_points, exaggeration);
    }

    void GpgpuSneVulkan::compute_sequence(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      auto range_x = abs(_bounds[2] - _bounds[0]);
      auto range_y = abs(_bounds[3] - _bounds[1]);
      // std::cout << "exaggeration: " << exaggeration << std::endl;

       //std::cout << "min x,y" << bounds[0] << "," << bounds[1] << std::endl;
       //std::cout << "max x,y" << bounds[2] << "," << bounds[3] << std::endl;
     // std::cout << "Range X: " << range_x << " Range Y: " << range_y << std::endl;

      // assume adaptive resolution(scales with points range) with a minimum size
      auto width = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_x, float(MINIMUM_FIELDS_SIZE))));
      auto height = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_y, float(MINIMUM_FIELDS_SIZE))));

      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();
      bool new_field_buf = false;
      if ((int)iteration == 0) { // only on the first iteration
        _tensors[ShaderBuffers::POSITION]->setData(embedding->getContainer());
        //_tensors[ShaderBuffers::NUM_POINTS]->setData(std::vector<unsigned int>({ num_points }));
        // on first iteration the bound were calculated on the CPU so load them to the tensor
        _tensors[ShaderBuffers::BOUNDS]->setData(_bounds);
        _fields_buffer_size = 8;
        new_field_buf = true;
      }
      else if (width > _fields_buffer_size || height > _fields_buffer_size) {
        if (width > 2048 || height > 2048) {
          throw std::runtime_error("Field size larger than 2048 not supported");
        }
        while (width > _fields_buffer_size || height > _fields_buffer_size)
          _fields_buffer_size = std::min(2 * _fields_buffer_size, 2048u);
        new_field_buf = true;
      }
      //else if (width < _fields_buffer_size / 4 && height < _fields_buffer_size / 4) {
      //  _fields_buffer_size = std::max(_fields_buffer_size/4, 8u);
      //  new_field_buf = true;
      //}

      auto tu0 = std::chrono::high_resolution_clock::now();
      if (new_field_buf) {
        std::cout << "New field size: " << _fields_buffer_size << " iter " << iteration << "\n";
        // rerecord the computer buffer sequence with the new field size
        ; // at most 1024 (should this be an exception?)
        record_compute_sequence(iteration, width, height, num_points, _bounds.data(), exaggeration, mult);
      } else {
        // simply update the push constants of the sequence
        update_compute_sequence(iteration, num_points, width, height, _bounds.data(), exaggeration, mult);
      }
      //auto tu1 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_tu = std::chrono::duration<double, std::milli>(tu1 - tu0).count();
      //auto t0 = std::chrono::high_resolution_clock::now();
      { _seq0->eval();}
      //auto t1 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_0 = std::chrono::duration<double, std::milli>(t1 - t0).count();
      //auto t2 = std::chrono::high_resolution_clock::now();
      //{ _seq1->eval(); }
      //auto t3 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_1 = std::chrono::duration<double, std::milli>(t3 - t2).count();
      //auto t4 = std::chrono::high_resolution_clock::now();
      //{ _seq2->eval();}
      //auto t5 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_2 = std::chrono::duration<double, std::milli>(t5 - t4).count();
      //auto t6 = std::chrono::high_resolution_clock::now();
      { _seq3->eval(); }
      //auto t7 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_3 = std::chrono::duration<double, std::milli>(t7 - t6).count();

      
      //_totalTime += cpu_ms_0 + cpu_ms_1 + cpu_ms_2 + cpu_ms_3 + cpu_ms_tu;
      //double texsize = (_bounds[2] - _bounds[0]) * (_bounds[3] - _bounds[1]);
      //double ms_per_texel = cpu_ms_0 / texsize;
      //+field=%.3f, interp field=%.3f, ---- cpu_ms_1, cpu_ms_2, 
      //printf("iter: %u, stencil=%.3f,  field=%.3f, interp field=%.3f, forces+disp=%.3f, total=%.3f, TexSize=%.0f, ms per texel=%0.7f \n", int(iteration), cpu_ms_0, cpu_ms_1, cpu_ms_2, cpu_ms_3, cpu_ms_tu, _totalTime, texsize, ms_per_texel);
      // for debug purposes only - get the values locally 
      auto syncSeq = _mgr->sequence();
      syncSeq->record<kp::OpSyncLocal>(std::vector<std::shared_ptr<kp::Memory>> {
        _tensors[ShaderBuffers::BOUNDS],
        _tensors[ShaderBuffers::POSITION],
        _tensors[ShaderBuffers::KLDIV],
        _tensors[ShaderBuffers::IMAGE_WORKGROUP],
        _tensors[ShaderBuffers::SUM_Q],/*
        _shaderImageHelper.getActivePixelList(),
        _shaderImageHelper.getStencilImage(),
        _tensors[ShaderBuffers::ATOMIC_COUNTER],
        _shaderImageHelper.getFieldImage(),
        _tensors[ShaderBuffers::SUM_Q],
        _tensors[ShaderBuffers::PARTIAL_SUM],
        _tensors[ShaderBuffers::INTERP_FIELDS],
        _tensors[ShaderBuffers::GRADIENTS],
        _tensors[ShaderBuffers::PREV_GRADIENTS],
        _tensors[ShaderBuffers::GAIN],
        _tensors[ShaderBuffers::DEBUG],*/
      })->eval();
      /*auto stencil = static_cast<kp::Image*>(_shaderImageHelper.getStencilImage().get())->vector<float>();
      auto field = static_cast<kp::Image*>(_shaderImageHelper.getFieldImage().get())->vector<float>();
      auto sum_q = _interpEnhProg->getSumQ();
      auto partial = _tensors[ShaderBuffers::PARTIAL_SUM]->vector<float>();
      auto interp_fields = _tensors[ShaderBuffers::INTERP_FIELDS]->vector<float>();
      auto grads = _tensors[ShaderBuffers::GRADIENTS]->vector<float>();
      auto prevGrads = _tensors[ShaderBuffers::PREV_GRADIENTS]->vector<float>();
      auto gain = _tensors[ShaderBuffers::PREV_GRADIENTS]->vector<float>();
      auto debug = _tensors[ShaderBuffers::DEBUG]->vector<float>();
      auto activeList = _shaderImageHelper.getActivePixelList()->vector();
      auto atom_counter = _tensors[ShaderBuffers::ATOMIC_COUNTER]->vector<uint32_t>()[0];*/
      auto sum_q = _interpEnhProg->getSumQ();
      auto wrkgrp = _tensors[ShaderBuffers::IMAGE_WORKGROUP]->vector<uint32_t>();
      //printf("Width %i height %i, Workgroups dispatched: x=%u, y=%u, z=%u\n", width, height, wrkgrp[0], wrkgrp[1], wrkgrp[2]);
      auto positions = _tensors[ShaderBuffers::POSITION]->vector<float>();
      _bounds = _tensors[ShaderBuffers::BOUNDS]->vector<float>();
      kl_divergence = _tensors[ShaderBuffers::KLDIV]->vector<float>()[0];
      std::cout << "sumq: " << sum_q << " kl_div: " << kl_divergence << "\n";
      if (kl_divergence < 0) {
        std::cout << "Sequence KL Divergence is negative, at iteration: " << iteration;
      }
      memcpy(points, positions.data(), 2*num_points*sizeof(float));

    }
  }
}