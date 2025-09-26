#include "gpgpu_sne_comp_vulkan.h"

#include <vector>
#include <limits> 
#include <iostream>
#include <cmath> // for sqrt
#include <memory>
#include <sstream>
#include "tensor_config.h"
#include "shaders/shaders.h"
#include "gpgpu_sne_comp_vulkan.h"

namespace hdi {
  namespace dr {
    typedef GpgpuSneVulkan::Bounds2D Bounds2D;
    typedef GpgpuSneVulkan::Point2D Point2D;

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

    Bounds2D GpgpuSneVulkan::computeInitialBounds(const embedding_type* embedding, float padding) {
      const float* const points = embedding->getContainer().data();

      Bounds2D bounds;
      bounds.min.x = std::numeric_limits<float>::max();
      bounds.max.x = -std::numeric_limits<float>::max();
      bounds.min.y = std::numeric_limits<float>::max();
      bounds.max.y = -std::numeric_limits<float>::max();

      for (int i = 0; i < embedding->numDataPoints(); ++i) {
        float x = points[i * 2 + 0];
        float y = points[i * 2 + 1];

        bounds.min.x = std::min<float>(x, bounds.min.x);
        bounds.max.x = std::max<float>(x, bounds.max.x);
        bounds.min.y = std::min<float>(y, bounds.min.y);
        bounds.max.y = std::max<float>(y, bounds.max.y);
      }

      // Add any extra padding if requested
      if (padding != 0) {
        float half_padding = padding / 2;

        float x_padding = (bounds.max.x - bounds.min.x) * half_padding;
        float y_padding = (bounds.max.y - bounds.min.y) * half_padding;

        bounds.min.x -= x_padding;
        bounds.max.x += x_padding;
        bounds.min.y -= y_padding;
        bounds.max.y += y_padding;
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
      _tensors[ShaderBuffers::POSITION] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::INTERP_FIELDS] = _mgr.tensorT(std::vector<float>(num_pnts * 4, 0.0f));
      _tensors[ShaderBuffers::SUM_Q] = _mgr.tensorT(std::vector<float>(1, 0.0f));
      _tensors[ShaderBuffers::KLDIV] = _mgr.tensorT(std::vector<float>(1, 0.0f));
      _tensors[ShaderBuffers::GRADIENTS] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::NEIGHBOUR] = _mgr.tensorT(linear_P.neighbours);
      _tensors[ShaderBuffers::PROBABILITIES] = _mgr.tensorT(linear_P.probabilities);
      _tensors[ShaderBuffers::INDEX] = _mgr.tensorT(linear_P.indices);
      _tensors[ShaderBuffers::PREV_GRADIENTS] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
      _tensors[ShaderBuffers::GAIN] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 1.0f));
      _tensors[ShaderBuffers::BOUNDS] = _mgr.tensorT(std::vector<float>(4, 1.0f));
      _tensors[ShaderBuffers::NUM_POINTS] = _mgr.tensorT(std::vector<unsigned int>(1, num_pnts));

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
      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();
      if (iteration < 0.5) { // only on the first iteration
        _tensors[ShaderBuffers::POSITION]->setData(embedding->getContainer());
        _tensors[ShaderBuffers::NUM_POINTS]->setData(std::vector<unsigned int>({ num_points }));
      }

      // Compute the bounds of the given embedding and add a 10% border around it
      auto padding = 0.1f;
      std::vector<float> bounds = _boundsProg->compute(padding);

      auto range_x = abs(bounds[2] - bounds[0]);
      auto range_y = abs(bounds[3] - bounds[1]);

      std::cout << "min x,y" << bounds[0] << "," << bounds[1] << std::endl;
      std::cout << "max x,y" << bounds[2] << "," << bounds[3] << std::endl;
      std::cout << "Range X: " << range_x << " Range_Y: " << range_y << std::endl;

      // assume adaptive resolution(scales with points range) with a minimum size
      auto width = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_x, float(MINIMUM_FIELDS_SIZE))));
      auto height = static_cast<uint32_t>(std::floor(std::max(RESOLUTION_SCALING * range_y, float(MINIMUM_FIELDS_SIZE))));

      // calculate the fields texture
      auto stencil = _stencilProg->compute(width, height, num_points, bounds);
      auto fields = _fieldCompProg->compute(stencil, width, height);

      // Calculate the normalization sum
      _interpProg->compute(fields, width, height);
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
      padding = 0.0f;
      bounds = _boundsProg->compute(padding);
      // Use that to center the embedding around 0,0 and scale it if needed
      auto positions = _centerScaleProg->compute(num_points, exaggeration);
      size_t size = sizeof(float) * 2 * num_points;

      // Save the new embedding
      //embedding->getContainer().assign(positions.data(), positions.data() + size);
      memcpy(points, positions.data(), size);
    }
  }
}