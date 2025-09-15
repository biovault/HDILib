// This file contains the implementation of the GPGPU SNE computation using Vulkan.
// The structure is similar to gpgpu_sne_compute.h but adapted for Vulkan API using the 
// vulkan kompute library.
//#ifdef USE_VULKAN_KOMPUTE 
#pragma once

#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"

#include <kompute/Kompute.hpp>
namespace hdi {
  namespace dr {
    struct LinearProbabilityMatrix;
    class GpgpuSneVulkan {
    public:     
        struct Point2D {
            float x, y;
        };
    
        struct Bounds2D {
            Point2D min;
            Point2D max;
    
            Point2D getRange() {
            return Point2D{ max.x - min.x, max.y - min.y };
            }
        };
    
        typedef hdi::data::Embedding<float> embedding_type;
        typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix_type;
        float kl_divergence = -1.0f; // Kullback-Leibler divergence, only computed in the compute shader version otherwise defaults to -1.0f

    public:
        GpgpuSneVulkan();

        void initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P);
        void clean();

        // a single iteration of the tSNE algorithm involving field computation, forces computation and embedding update
        void compute(embedding_type* embedding, float exaggeration, float iteration, float mult);

        void setScalingFactor(float factor) { _resolutionScaling = factor; }
        //!  Change the runtime configurable params
        void updateParams(TsneParameters params) { 
            if (!_initialized) {
                throw std::runtime_error("GradientDescentComputation must be initialized before updating the tsne parameters");
            }
            _params = params;
        }
        bool isInitialized() { return _initialized == true; }
    private:
        const unsigned int FIXED_FIELDS_SIZE = 40;
        const unsigned int MINIMUM_FIELDS_SIZE = 5;
        const float PIXEL_RATIO = 2;

        void initializeVulkan(unsigned int num_pnts, const LinearProbabilityMatrix& linear_P);

        Bounds2D computeInitialBounds(const embedding_type* embedding, float padding);
        void calculateNormalizationFactor(float* Z_hat); 
        void computeGradients(unsigned int num_points, float Z_hat, float exaggeration);
        void updatePoints(unsigned int num_points, float* points, embedding_type* embedding, float iteration, float mult);
        void updateEmbedding(unsigned int num_points, float exaggeration, float iteration, float mult);


        bool _initialized = false;
        float _resolutionScaling = 1.0f;
        TsneParameters _params;
        bool _adaptive_resolution;
        // Embedding bounds
        Bounds2D _bounds;

        kp::Manager _mgr;

        enum Tensors {
            POSITION = 0,
            NEIGHBOUR = 1,
            PROBABILITIES = 2,
            INDEX = 3,
            INTERP_FIELDS = 4,
            GRADIENTS = 5,
            PREV_GRADIENTS = 6,
            GAIN = 7,
            BOUNDS = 8,
            SUM_Q = 9,
            KLDIV = 10,
            NUM_POINTS = 11
        };

        std::map<Tensors, std::shared_ptr<kp::Tensor>> _tensors;

    };
}
}

//#endif