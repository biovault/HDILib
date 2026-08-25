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
      // for performance reasons kl_divergence calculation is disabled.
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
      //if (num_points < 1000) {
      //  _resolutionScaling = 8;
      //}

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
      std::cout << " Initial bounds: " << _bounds[0] << ", " << _bounds[1] << ", " << _bounds[2] << ", "  << _bounds[3] << "\n";

      // Initialize all Vulkan resources
      initializeVulkan(num_points, linear_P);
      std::cout << "Vulkan set initialize /n";
      _initialized = true;
    }

    void GpgpuSneVulkan::initializeVulkan(unsigned int num_pnts, const LinearProbabilityMatrix& linear_P) {
      // Create the manager with debug extensions
#ifdef __APPLE__
      _mgr = std::make_shared<kp::Manager>(0, std::vector<uint32_t>(), std::vector<std::string>({
        "VK_KHR_synchronization2",
        "VK_KHR_portability_subset",
        "VK_EXT_metal_objects"}));
#else
      _mgr = std::make_shared<kp::Manager>(0, std::vector<uint32_t>(), std::vector<std::string>({
        "VK_KHR_synchronization2",
        "VK_KHR_shader_float_controls2" }));
    #endif

      _tensors[ShaderBuffers::POSITION] = _mgr->tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::INTERP_FIELDS] = _mgr->tensorT(std::vector<float>(num_pnts * 4, 0.0f));
      _tensors[ShaderBuffers::SUM_Q] = _mgr->tensorT(std::vector<float>(1, 0.0f));
      _tensors[ShaderBuffers::KLDIV] = _mgr->tensorT(std::vector<float>(1, -1.0f));
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
      //std::cout << "Interoplate using " << _numInterpWorkgroups << " workgroups\n";
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
      _tensors[ShaderBuffers::UBO_FIELD_PF_1] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(fieldParamsPF1), 0));
      _tensors[ShaderBuffers::UBO_FIELD_PF_2] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(fieldParamsPF2), 0));


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

      // Upload the probability values and indices to the GPU once
      auto seq = _mgr->sequence();
      const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
        _tensors[ShaderBuffers::NEIGHBOUR],
        _tensors[ShaderBuffers::INDEX],
        _tensors[ShaderBuffers::PROBABILITIES],
      };
      seq->record<kp::OpSyncDevice>(syncParams);
      seq->eval();

    }

    void GpgpuSneVulkan::clean() {
      for (auto& n : _tensors)
        n.second->destroy();
      _tensors.clear();
      std::cout << "Vulkan in clear initialize /n";
      _initialized = false;
      
      //std::cout << " Clearing vulkan resources\n";
      auto sten = _shaderImageHelper.getStencilImage();
      auto fiel = _shaderImageHelper.getFieldImage();
      auto samp = _shaderImageHelper.getFieldSamplerImage();
      //std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
      
      if (fiel.lock()) {
        auto field_vkimage = static_cast<VkImage>(*fiel.lock()->getPrimaryImage().get());
        auto stencil_vkimage = static_cast<VkImage>(*sten.lock()->getPrimaryImage().get());
        auto sample_vkimage = static_cast<VkImage>(*samp.lock()->getPrimaryImage().get());
        //std::cout << " VkImages -  field: " << field_vkimage << " sample: " << sample_vkimage << " stencil: " << stencil_vkimage << "\n";
      }
      
      _stencilProg.reset();
      _stencil2ListProg.reset();
      _fieldCompEnhProg.reset();
      _interpEnhProg.reset();
      _seq0.reset();
      //std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
      _shaderImageHelper.resetBuffers();
      //std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
    }

    void GpgpuSneVulkan::compute(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      compute_sequence(embedding, exaggeration, iteration, mult);
    }

    void GpgpuSneVulkan::record_compute_sequence(
      float iteration,
      uint32_t width,
      uint32_t height,
      uint32_t num_points,
      float* bounds,
      float exaggeration,
      float mult) {
      
      auto sten = _shaderImageHelper.getStencilImage();
      auto fiel = _shaderImageHelper.getFieldImage();
      auto samp = _shaderImageHelper.getFieldSamplerImage();
      if (_seq0) {
        if (fiel.lock()) {
          auto field_vkimage = static_cast<VkImage>(*fiel.lock()->getPrimaryImage().get());
          auto stencil_vkimage = static_cast<VkImage>(*sten.lock()->getPrimaryImage().get());
          auto sample_vkimage = static_cast<VkImage>(*samp.lock()->getPrimaryImage().get());
          //std::cout << " VkImages -  field: " << field_vkimage << " sample: " << sample_vkimage << " stencil: " << stencil_vkimage << "\n";
        }
        // free all old VULKAN image resources
        // These will be recreated at the new size
        //std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
        _stencilProg = std::make_shared<StencilShaderProg>(_mgr, _tensors);
        _stencil2ListProg = std::make_shared<Stencil2ListShaderProg>(_mgr, _tensors);
        _fieldCompEnhProg = std::make_shared<FieldComputationEnhShaderProg>(_mgr, _tensors);
        //_fieldCompPFProg = std::make_shared<FieldComputationPointFirstShaderProg>(_mgr, _tensors);
        //_interpEnhProg = std::make_shared<InterpolationEnhShaderProg>(_mgr, _tensors);
        _shaderImageHelper.resetBuffers();
        _seq0.reset();

        //std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
      }
      _seq0 = _mgr->sequence();
      //if (sten.lock())
      //  std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
      _shaderImageHelper.createBuffers(_mgr, _fields_buffer_size, num_points, 256);
      _shaderImageHelper.setFieldArraySampler(_mgr->createLinearSampler());
      //if (sten.lock())
      //  std::cout << "Use counts - stencil : " << sten.use_count() << " field: " << fiel.use_count() <<  " samp: " << samp.use_count() << "\n";
      _seq0->begin();
      //_shaderImageHelper.resetBuffers();

      _stencilProg->record(_seq0, width, height, _shaderImageHelper.getStencilImage(), num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _stencil2ListProg->record(_seq0, _fields_buffer_size, _fields_buffer_size, _shaderImageHelper.getStencilImage(), _shaderImageHelper.getActivePixelList(), num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _fieldCompEnhProg->record(_seq0, num_points, width, height, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), _shaderImageHelper.getActivePixelList(), _fields_buffer_size);
      //_fieldCompPFProg->record(_seq0, num_points, width, height, _shaderImageHelper.getNumWorkgroups(), _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), _shaderImageHelper.getActivePixelList(), _shaderImageHelper.getPartialResults(), _fields_buffer_size);
      _interpEnhProg->record(_seq0, num_points, _numInterpWorkgroups, _shaderImageHelper.getFieldSamplerImage(), _shaderImageHelper.getFieldImage(), width, height);
      _seq0->end();
      if (_seq1.get() == nullptr) {
        _seq1 = _mgr->sequence();
        _seq1->begin();
        _forcesProg->record(_seq1, num_points, exaggeration);
        _updateProg->record(_seq1, num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
        _boundsProg->record_unpadded(_seq1, num_points);
        _centerScaleProg->record(_seq1, num_points, exaggeration);
        _boundsProg->record_padded(_seq1, num_points, 0.1f);
        _seq1->end();
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
      _fieldCompEnhProg->update(num_points, width, height, _fields_buffer_size);
      //_fieldCompPFProg->update(num_points, width, height, _shaderImageHelper.getNumWorkgroups(), _fields_buffer_size);
      _interpEnhProg->update(num_points, width, height);
      _forcesProg->update(num_points, exaggeration);
      _updateProg->update(num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
      _centerScaleProg->update(num_points, exaggeration);
    }

    void GpgpuSneVulkan::compute_sequence(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      //std::cout << "Bounds size: " << _bounds.size();
      auto range_x = abs(_bounds[2] - _bounds[0]);
      auto range_y = abs(_bounds[3] - _bounds[1]);

      // assume adaptive resolution(scales with points range) with a minimum size
      auto width = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_x, float(MINIMUM_FIELDS_SIZE))));
      auto height = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_y, float(MINIMUM_FIELDS_SIZE))));

      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();
      bool new_field_buf = false;
      if ((int)iteration == 0) { // only on the first iteration
        // on the first iteration we get the starting embedding positions from the CPU
        _tensors[ShaderBuffers::POSITION]->setData(embedding->getContainer());
        // on first iteration the bound were calculated on the CPU so load them to the tensor
        _tensors[ShaderBuffers::BOUNDS]->setData(_bounds);
        // Upload the values to the GPU
        auto seq = _mgr->sequence();
        const std::vector<std::shared_ptr<kp::Memory>> syncParams = {
          _tensors[ShaderBuffers::POSITION],
          _tensors[ShaderBuffers::BOUNDS]
        };
        seq->record<kp::OpSyncDevice>(syncParams);
        seq->eval();
        _fields_buffer_size = 32;
        new_field_buf = true;
      }
      else if (width > _fields_buffer_size || height > _fields_buffer_size) {
        if (width > 2048 || height > 2048) {
          // for GPU capture do early return
          return;
          //throw std::runtime_error("Field size larger than 2048 not supported");
        }
        while (width > _fields_buffer_size || height > _fields_buffer_size)
          _fields_buffer_size = std::min(32 + _fields_buffer_size, 2048u);
        new_field_buf = true;
      }
      /*else if (width < _fields_buffer_size - 32 && height < _fields_buffer_size - 32 && _fields_buffer_size >= 64) {
        // support shrinking for performance reasons
        _fields_buffer_size = _fields_buffer_size - 32;
        new_field_buf = true;
      }      */



      //auto tu0 = std::chrono::high_resolution_clock::now();
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
      { _seq1->eval(); }
      //auto t3 = std::chrono::high_resolution_clock::now();
      //double cpu_ms_1 = std::chrono::duration<double, std::milli>(t3 - t2).count();

      //*** DEBUG ****/
      /*/auto syncSeq = _mgr->sequence();
      syncSeq->record<kp::OpSyncLocal>(std::vector<std::shared_ptr<kp::Memory>> {
        _shaderImageHelper.getFieldImage().lock(),
        _shaderImageHelper.getStencilImage().lock(),
        _shaderImageHelper.getActivePixelList().lock(),
        _shaderImageHelper.getPartialResults().lock(),
        _shaderImageHelper.getActivePixelList().lock(),
        _tensors[ShaderBuffers::ATOMIC_COUNTER],
        _tensors[ShaderBuffers::SUM_Q],
        _tensors[ShaderBuffers::PARTIAL_SUM],
        _tensors[ShaderBuffers::INTERP_FIELDS],
        _tensors[ShaderBuffers::GRADIENTS],
        _tensors[ShaderBuffers::PREV_GRADIENTS],
        _tensors[ShaderBuffers::GAIN],
        _tensors[ShaderBuffers::DEBUG],
      })->eval();

      auto stencil = static_cast<kp::Image*>(_shaderImageHelper.getStencilImage().lock().get())->vector<float>();
      auto field = static_cast<kp::Image*>(_shaderImageHelper.getFieldImage().lock().get())->vector<float>();
      auto partial = static_cast<kp::TensorT<float>*>(_shaderImageHelper.getPartialResults().lock().get())->data();
      auto active_pixels = static_cast<kp::TensorT<uint32_t>*>(_shaderImageHelper.getActivePixelList().lock().get())->data();
      auto counter = _tensors[ShaderBuffers::ATOMIC_COUNTER]->vector<uint32_t>()[0];
      auto sum_q = _tensors[ShaderBuffers::SUM_Q]->vector<float>()[0];
      auto interp_fields = _tensors[ShaderBuffers::INTERP_FIELDS]->vector<float>();
      auto grads = _tensors[ShaderBuffers::GRADIENTS]->vector<float>();*/

      //*** END DEBUG ****/

      auto positions = _tensors[ShaderBuffers::POSITION]->vector<float>();
      _bounds = _tensors[ShaderBuffers::BOUNDS]->vector<float>();
      //kl_divergence = _tensors[ShaderBuffers::KLDIV]->vector<float>()[0];
      memcpy(points, positions.data(), 2*num_points*sizeof(float));

    }
  }
}
