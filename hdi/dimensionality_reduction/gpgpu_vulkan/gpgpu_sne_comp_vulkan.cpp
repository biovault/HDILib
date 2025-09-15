#include "gpgpu_sne_comp_vulkan.h"

#include <vector>
#include <limits> 
#include <iostream>
#include <cmath> // for sqrt

namespace hdi {
  namespace dr {
    typedef GpgpuSneVulkan::Bounds2D Bounds2D;
    typedef GpgpuSneVulkan::Point2D Point2D;

    // Linearized sparse neighbourhood matrix
    struct LinearProbabilityMatrix
    {
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
      initializeVulkan(num_pts, linear_P);

      _initialized = true;
    }

    GpgpuSneVulkan::initializeVulkan(unsigned int num_pnts, const LinearProbabilityMatrix& linear_P) {
        _tensors[POSITION] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
        _tensors[INTERP_FIELDS] = _mgr.tensorT(std::vector<float>(num_pnts * 4, 0.0f));
        _tensors[SUM_Q] = _mgr.tensorT(std::vector<float>(1, 0.0f));
        _tensors[KLDIV] = _mgr.tensorT(std::vector<float>(1, 0.0f));
        _tensors[GRADIENTS] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
        _tensors[NEIGHBOUR] = _mgr.tensorT(linear_P.neighbours);
        _tensors[PROBABILITIES] = _mgr.tensorT(linear_P.probabilities);
        _tensors[INDEX] = _mgr.tensorT(linear_P.indices);
        _tensors[PREV_GRADIENTS] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 0.0f));
        _tensors[GAIN] = _mgr.tensorT(std::vector<float>(num_pnts * 2, 1.0f));
        _tensors[BOUNDS] = _mgr.tensorT(std::vector<float>{std::vector<float>(4, 1.0f)});
        _tensors[NUM_POINTS] = _mgr.tensorT(std::vector<int>{static_cast<uint32_t>(1, num_pnts)});
    }
}