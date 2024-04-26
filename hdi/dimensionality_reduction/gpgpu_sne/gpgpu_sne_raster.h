/*
*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
* EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
*
*/

#pragma once

#ifdef __APPLE__
    #include <OpenGL/gl3.h>
#else // __APPLE__
    #include "hdi/utils/glad/glad.h"
#endif // __APPLE__

#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "field_computation.h"

#include <array>
#include <cstdint>

namespace hdi {
  namespace dr {
    template<typename T, typename S>
    struct LinearProbabilityMatrix;

    //! Computation class for texture-based t-SNE using rasterization
    /*!
    Computation class for texture-based t-SNE using rasterization
    \author Julian Thijssen
    */
    template <typename unsigned_integer = std::uint32_t>
    class GpgpuSneRaster {
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

      using unsigned_int_type = unsigned_integer;
      using int_type = typename std::conditional<std::is_same_v<unsigned_integer, std::uint32_t>, std::int32_t, std::int64_t>::type;
      using embedding_type = hdi::data::Embedding<float>;
      using sparse_scalar_matrix_type = std::vector<hdi::data::MapMemEff<unsigned_int_type, float>>;

    public:
      GpgpuSneRaster();

      void initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P);
      void clean();

      void compute(embedding_type* embedding, float exaggeration, float iteration, float mult);

      void setScalingFactor(float factor) { _resolutionScaling = factor; }
	  //!  Change the runtime configurable params
	  void updateParams(TsneParameters params) {
		  if (!_initialized) {
			  throw std::runtime_error("GradientDescentComputation must be initialized before updating the tsne parameters");
		  }
		  _params = params;
	  };
	  bool isInitialized() { return _initialized == true; }
    private:
      void initializeOpenGL(const unsigned_int_type num_points, const LinearProbabilityMatrix<unsigned_int_type, int_type>& linear_P);

      Bounds2D computeEmbeddingBounds(const embedding_type* embedding, float padding = 0);

      void interpolateFields(unsigned_int_type num_points, unsigned int width, unsigned int height);
      void computeGradients(embedding_type* embedding, unsigned_int_type num_points, double exaggeration);
      void updateEmbedding(embedding_type* embedding, float exaggeration, float iteration, float mult);

    private:
      const unsigned int FIXED_FIELDS_SIZE = 40;
      const unsigned int MINIMUM_FIELDS_SIZE = 5;
      const float PIXEL_RATIO = 2;

      bool _initialized = false;
      bool _adaptive_resolution = true;

      float _resolutionScaling = PIXEL_RATIO;

      GLuint _dummy_fbo = {};
      GLuint _dummy_tex = {};

      // Shaders
      ShaderProgram _interp_program = {};

      // Buffers
      GLuint _dummy_vao = {};
      GLuint _position_buffer = {};

      std::vector<double> _positive_forces = {};
      std::vector<double> _negative_forces = {};

      // Gradient descent
      std::vector<double> _gradient = {};
      std::vector<double> _previous_gradient = {};
      std::vector<double> _gain = {};

      std::vector<float> _interpolated_fields = {};
      sparse_scalar_matrix_type _P = {};

      RasterFieldComputation fieldComputation = {};

      // Embedding bounds
      Bounds2D _bounds = {};

      // T-SNE parameters
      TsneParameters _params = {};

      // Probability distribution function support 
      float _function_support = 0;
    };
  }
}
