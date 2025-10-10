// This file contains the implementation of the GPGPU SNE computation using Vulkan.
// The structure is similar to gpgpu_sne_compute.h but adapted for Vulkan API using the 
// vulkan kompute library.
//#ifdef USE_VULKAN_KOMPUTE 
#pragma once

#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "tensor_config.h"
#include "shaders/shader_progs.h"

#include <kompute/Kompute.hpp>
namespace hdi {
  namespace dr {
    struct LinearProbabilityMatrix;
    class GpgpuSneVulkan {
    public:

      typedef hdi::data::Embedding<float> embedding_type;
      typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix_type;
      float kl_divergence = -1.0f; // Kullback-Leibler divergence, only computed in the compute shader version otherwise defaults to -1.0f

    public:
      GpgpuSneVulkan();

      void initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P);
      void clean();

      // a single iteration of the tSNE algorithm involving field computation, forces computation and embedding update
      void compute(embedding_type* embedding, float exaggeration, float iteration, float mult);
      void compute_stepwise(embedding_type* embedding, float exaggeration, float iteration, float mult);
      void compute_sequence(embedding_type* embedding, float exaggeration, float iteration, float mult);

      void setScalingFactor(float factor) { _resolutionScaling = factor; }
      //!  Change the runtime configurable params
      void updateParams(TsneParameters params) {
        if (!_initialized) {
          throw std::runtime_error("GradientDescentComputation must be initialized before updating the tsne parameters");
        }
        _params = params;
      }
      bool isInitialized() const { return _initialized == true; }

    private:
      const unsigned int FIXED_FIELDS_SIZE = 40;
      const unsigned int MINIMUM_FIELDS_SIZE = 5;
      const float PIXEL_RATIO = 2;
      const float RESOLUTION_SCALING = 2.0;

      void initializeVulkan(unsigned int num_pnts, const LinearProbabilityMatrix& linear_P);

      std::vector<float> computeInitialBounds(const embedding_type* embedding, float padding);
      void record_compute_sequence(uint32_t width, uint32_t height, uint32_t num_points, float* bounds);
      void update_compute_sequence(uint32_t width, uint32_t height, float* bounds, float exaggeration);

      bool _initialized = false;
      float _resolutionScaling = 1.0f;
      TsneParameters _params;
      bool _adaptive_resolution;
      // Embedding bounds
      std::vector<float> _bounds;
      unsigned int _fields_buffer_size;

      // kompute manager context
      std::shared_ptr <kp::Manager> _mgr;
      // recorded sequence for the compute shader version
      std::shared_ptr<kp::Sequence> _seq;
      // kompute tensor buffers
      std::map<ShaderBuffers, std::shared_ptr<kp::Tensor>> _tensors;
      // kompute compute shaders
      std::shared_ptr<BoundsShaderProg> _boundsProg;
      std::shared_ptr<StencilShaderProg> _stencilProg;
      std::shared_ptr<FieldComputationShaderProg> _fieldCompProg;
      std::shared_ptr<InterpolationShaderProg> _interpProg;
      std::shared_ptr<ForcesShaderProg> _forcesProg;
      std::shared_ptr<UpdateShaderProg> _updateProg;
      std::shared_ptr<CenterScaleShaderProg> _centerScaleProg;
    };
  }
}

//#endif