#include "gpgpu_sne_comp_vulkan.h"

#include <vector>
#include <limits> 
#include <iostream>
#include <cmath> // for sqrt
#include <memory>
#include <sstream>
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
      _resolutionScaling(PIXEL_RATIO),
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
      _mgr = std::make_shared<kp::Manager>(0); //"VK_LAYER_NV_nsight", "VK_LAYER_KHRONOS_validation"  "VK_LAYER_NV_GPU_Trace_release_public_2025_4_1"
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
#ifndef SHADER_USE_PUSH_CONSTANTS
      // These tensors will be used as UBOs
      _tensors[ShaderBuffers::UBO_STENCIL] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(stencilParams), 0));
      _tensors[ShaderBuffers::UBO_FIELD] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(fieldParams), 0));
      _tensors[ShaderBuffers::UBO_INTERP] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(interpParams), 0));
      _tensors[ShaderBuffers::UBO_FORCES] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(forcesParams), 0));
      _tensors[ShaderBuffers::UBO_UPDATE] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(updaterParams), 0));
      _tensors[ShaderBuffers::UBO_CENTER_SCALE] = _mgr->uboTensorT<uint8_t>(std::vector<uint8_t>(sizeof(centerScaleParams), 0));
#endif

      _boundsProg = std::make_shared<BoundsShaderProg>(_mgr, _tensors);
      _stencilProg = std::make_shared<StencilShaderProg>(_mgr, _tensors);
      _fieldCompProg = std::make_shared<FieldComputationShaderProg>(_mgr, _tensors);
      _interpProg = std::make_shared<InterpolationShaderProg>(_mgr, _tensors);
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

    void GpgpuSneVulkan::compute_stepwise(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      auto range_x = abs(_bounds[2] - _bounds[0]);
      auto range_y = abs(_bounds[3] - _bounds[1]);
      // std::cout << "exaggeration: " << exaggeration << std::endl;

       //std::cout << "min x,y" << bounds[0] << "," << bounds[1] << std::endl;
       //std::cout << "max x,y" << bounds[2] << "," << bounds[3] << std::endl;
      std::cout << "Range X: " << range_x << " Range Y: " << range_y << std::endl;

      // assume adaptive resolution(scales with points range) with a minimum size
      auto width = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_x, float(MINIMUM_FIELDS_SIZE))));
      auto height = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_y, float(MINIMUM_FIELDS_SIZE))));

      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();
      if (iteration < 0.5) { // only on the first iteration
        _tensors[ShaderBuffers::POSITION]->setData(embedding->getContainer());
        _tensors[ShaderBuffers::NUM_POINTS]->setData(std::vector<unsigned int>({ num_points }));
        // on first iteration the bound were calculated on the CPU so load them to the tensor
        _tensors[ShaderBuffers::BOUNDS]->setData(_bounds);
      }

      // calculate the fields texture
      auto stencil = _stencilProg->compute(width, height, num_points, _bounds);
      auto stencilData = stencil->data<float>();
      auto fields = _fieldCompProg->compute(stencil, width, height);
      auto fieldsData = fields->data<float>();

      // Calculate the normalization sum
      _interpProg->compute(fields, width, height);
      //std::cout << "width: " << width << " height: " << height << std::endl;
      if (_interpProg->getSumQ() == 0) {
        std::stringstream ss;
        ss << "Interpolation SumQ is 0, breaking out at iteration: " << iteration;
        throw std::runtime_error(ss.str());
        return;
      }

      // compute the forces based on the gradients record the kl_divergence
      kl_divergence = _forcesProg->compute(num_points, exaggeration);

      // Update the embedding positions
      _updateProg->compute(num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);

      // Get the bounds of the new embedding
      auto padding = 0.0f;
      _boundsProg->compute(padding);
      // Use that to center the embedding around 0,0 and scale it if needed
      auto positions = _centerScaleProg->compute(num_points, exaggeration);
      size_t size = sizeof(float) * 2 * num_points;

      // Recompute the bounds of the centered embedding
      // padded and save it for the following iteration
      padding = 0.1f;
      _bounds = _boundsProg->compute(padding);

      // Save the new embedding
      //embedding->getContainer().assign(positions.data(), positions.data() + size);
      memcpy(points, positions.data(), size);
    }

    void GpgpuSneVulkan::record_compute_sequence(
      float iteration, 
      uint32_t width, 
      uint32_t height, 
      uint32_t num_points, 
      float* bounds, 
      float exaggeration, 
      float mult) {
      // for testing only - record stencil only and eval it later
      //_stencilSeq = _mgr->sequence();
      //_stencilSeq->begin();
      //_stencilProg->record(_stencilSeq, width, height, num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      //_stencilSeq->end();

      _seq = _mgr->sequence();
      _seq->begin();
      _stencilProg->record(_seq, width, height, num_points, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _fieldCompProg->record(_seq, _stencilProg->_stencil_out, width, height, _fields_buffer_size);
      _interpProg->record(_seq, _fieldCompProg->_field_out, width, height);
      _forcesProg->record(_seq, num_points, exaggeration);
      _updateProg->record(_seq, num_points, _params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
      _boundsProg->record_unpadded(_seq);
      _centerScaleProg->record(_seq, num_points, exaggeration);
      _boundsProg->record_padded(_seq, 0.1f);
      _seq->end();
    }

    void GpgpuSneVulkan::update_compute_sequence(
      float iteration, 
      uint32_t width, 
      uint32_t height, 
      float* bounds, 
      float exaggeration, 
      float mult) {
      _stencilProg->update(width, height, std::vector<float>(bounds, bounds + 4), _fields_buffer_size);
      _fieldCompProg->update(width, height, _fields_buffer_size);
      _interpProg->update(width, height);
      _forcesProg->update(exaggeration);
      _updateProg->update(_params._eta, _params._minimum_gain, iteration, _params._momentum, _params._mom_switching_iter, _params._final_momentum, mult);
      _centerScaleProg->update(exaggeration);
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
      if (iteration < 0.5) { // only on the first iteration
        _tensors[ShaderBuffers::POSITION]->setData(embedding->getContainer());
        _tensors[ShaderBuffers::NUM_POINTS]->setData(std::vector<unsigned int>({ num_points }));
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
      else if (width < _fields_buffer_size / 2 && height < _fields_buffer_size / 2) {
        _fields_buffer_size = std::max(_fields_buffer_size/2, 8u);
        new_field_buf = true;
      }

      if (new_field_buf) {
        // rerecord the computer buffer sequence with the new field size
        ; // at most 1024 (should this be an exception?)
        record_compute_sequence(iteration, width, height, num_points, _bounds.data(), exaggeration, mult);
      } else {
        // simply update the push constants of the sequence
        update_compute_sequence(iteration, width, height, _bounds.data(), exaggeration, mult);
      }
     // _stencilSeq->eval();
      _seq->eval();
      /*
      auto stencil = static_cast<kp::Image*>(_stencilProg->_stencil_out.get())->vector<float>();
      auto field = static_cast<kp::Image*>(_fieldCompProg->_field_out.get())->vector<float>();
      auto sum_q = _interpProg->getSumQ();
      auto positions = _tensors[ShaderBuffers::POSITION]->vector<float>();
      auto interp_fields = _tensors[ShaderBuffers::INTERP_FIELDS]->vector<float>();
      */
      auto positions = _tensors[ShaderBuffers::POSITION]->vector<float>();
      _bounds = _tensors[ShaderBuffers::BOUNDS]->vector<float>();
      kl_divergence = _tensors[ShaderBuffers::KLDIV]->vector<float>()[0];
      if (kl_divergence < 0) {  
        std::cout << "Sequence KL Divergence is negative, at iteration: " << iteration;
      }
      memcpy(points, positions.data(), 2*num_points*sizeof(float));

    }
  }
}