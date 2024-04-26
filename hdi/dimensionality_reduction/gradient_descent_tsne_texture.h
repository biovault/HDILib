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

#ifndef GRADIENT_DESCENT_TSNE_TEXTURE_H
#define GRADIENT_DESCENT_TSNE_TEXTURE_H

#include "hdi/utils/assert_by_exception.h"
#include "hdi/utils/abstract_log.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "gpgpu_sne/gpgpu_sne_compute.h"
#include "gpgpu_sne/gpgpu_sne_raster.h"
#include "tsne_parameters.h"

#include <array>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace hdi {
  namespace dr {
    //! tSNE with sparse and user-defined probabilities
    /*!
    Implementation of the tSNE algorithm with sparse and user-defined probabilities
    \author Nicola Pezzotti
    */

    template <typename scalar = float, typename sparse_scalar_matrix = std::vector<hdi::data::MapMemEff<std::uint32_t, float> >, typename unsigned_integer = std::uint64_t, typename integer = std::int64_t>
    class GradientDescentTSNETexture {
    public:
#ifndef __APPLE__
      typedef enum { RASTER, COMPUTE_SHADER, AUTO_DETECT } GpgpuSneType;
#endif
      typedef scalar scalar_type;
      typedef unsigned_integer unsigned_int_type;
      typedef integer int_type;
      typedef sparse_scalar_matrix sparse_scalar_matrix_type;
      typedef std::vector<scalar_type> scalar_vector_type;              // Vector of scalar_type
      typedef typename sparse_scalar_matrix_type::value_type map_type;  // default hdi::data::MapMemEff
      typedef typename map_type::key_type map_key_type;                 // default std::uint32_t
      typedef typename map_type::mapped_type map_value_type;            // default float

    public:
      GradientDescentTSNETexture();

#ifndef __APPLE__
      //! Override the default compute type.
      void setType(GpgpuSneType _tsneType);
#endif
      //! Initialize the class with a list of distributions. A joint-probability distribution will be computed as in the tSNE algorithm
      void initialize(const sparse_scalar_matrix_type& probabilities, data::Embedding<scalar_type>* embedding, TsneParameters params = TsneParameters());
      //! Initialize the class with a joint-probability distribution. Note that it must be provided non initialized and with the weight of each row equal to 2.
      void initializeWithJointProbabilityDistribution(const sparse_scalar_matrix_type& distribution, data::Embedding<scalar_type>* embedding, TsneParameters params = TsneParameters());
      //! Change the runtime modifiable tsne parameters
      void updateParams(TsneParameters params);
      //! Reset the internal state of the class but it keeps the inserted data-points
      void reset();
      //! Reset the class and remove all the data points
      void clear();

      //! Get the position in the embedding for a data point
      void getEmbeddingPosition(scalar_vector_type& embedding_position, map_key_type handle)const;

      //! Get the number of data points
      size_t getNumberOfDataPoints() { return _P.size(); }
      //! Get P
      const sparse_scalar_matrix_type& getDistributionP() const { return _P; }
      //! Get Q
      const scalar_vector_type& getDistributionQ() const { return _Q; }

      //! Return the current log
      utils::AbstractLog* logger()const { return _logger; }
      //! Set a pointer to an existing log
      void setLogger(utils::AbstractLog* logger) { _logger = logger; }

      //! Do an iteration of the gradient descent
      void doAnIteration(double mult = 1);
      //! Compute the Kullback Leibler divergence
      double computeKullbackLeiblerDivergence();

      //! Set the current iterations
      void setIteration(unsigned int iteration) { _iteration = iteration; }
      //! iterations performed by the algo
      unsigned int iteration()const { return _iteration; }

      //! Set the adaptive texture scaling
      void setResolutionFactor(float factor) {
#ifndef __APPLE__
        if (_gpgpu_type == COMPUTE_SHADER)
          _gpgpu_compute_tsne.setScalingFactor(factor);
        else
          _gpgpu_raster_tsne.setScalingFactor(factor);
#else
        _gpgpu_raster_tsne.setScalingFactor(factor);
#endif
      }
      
      bool isInitialized() const { return _initialized == true; }

      //! Exageration baseline
      double& exaggeration_baseline() { return _exaggeration_baseline; }
      const double& exaggeration_baseline()const { return _exaggeration_baseline; }

    private:
      //! Compute High-dimensional distribution
      void computeHighDimensionalDistribution(const sparse_scalar_matrix_type& probabilities);
      //! Initialize the point in the embedding
      void initializeEmbeddingPosition(int seed, double multiplier = .1);
      //! Compute tSNE gradient with the texture based algorithm
      void doAnIterationImpl(double exaggeration);

      //! Compute the exaggeration factor based on the current iteration
      double exaggerationFactor() const;

    private:
      data::Embedding<scalar_type>* _embedding; //! embedding
      typename data::Embedding<scalar_type>::scalar_vector_type* _embedding_container;
      bool _initialized; //! Initialization flag

      double _exaggeration_baseline;

      sparse_scalar_matrix_type _P; //! Conditional probalility distribution in the High-dimensional space
      scalar_vector_type _Q; //! Conditional probalility distribution in the Low-dimensional space
      scalar_type _normalization_Q; //! Normalization factor of Q - Z in the original paper

#ifndef __APPLE__
      GpgpuSneCompute<map_key_type> _gpgpu_compute_tsne;
      GpgpuSneType _gpgpu_type;
#endif // __APPLE__
      GpgpuSneRaster<map_key_type> _gpgpu_raster_tsne;

      TsneParameters _params;
      unsigned int _iteration;

      utils::AbstractLog* _logger;
    };
  }
}
#endif
