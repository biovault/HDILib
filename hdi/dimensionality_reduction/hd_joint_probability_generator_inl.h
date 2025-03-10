/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
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


#ifndef HD_JOINT_PROBABILITY_GENERATOR_INL
#define HD_JOINT_PROBABILITY_GENERATOR_INL

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/dimensionality_reduction/knn_utils.h"

#include <numeric>
#include <random>

#ifdef _OPENMP
#include "omp.h"
#endif

#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif

namespace hdi {
  namespace dr {
    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Parameters::Parameters() :
      _perplexity(30),
      _perplexity_multiplier(3),
      _num_trees(4),
      _num_checks(1024),
      _aknn_algorithm(hdi::dr::KNN_FLANN),
      _aknn_metric(hdi::dr::KNN_METRIC_EUCLIDEAN),
      _aknn_algorithmP1(16), // default parameter for HNSW
      _aknn_algorithmP2(200) // default parameter for HNSW
    {
    }

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::Statistics() :
      _total_time(0),
      _distribution_time(0),
      _knnStatistics()
    {}

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::reset() {
      _total_time = 0;
      _knnStatistics._trees_construction_time = 0;
      _knnStatistics._aknn_time = 0;
      _distribution_time = 0;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::log(utils::AbstractLog* logger)const {
      utils::secureLog(logger, "\n-------- HD Joint Probability Generator Statistics -----------");
      utils::secureLogValue(logger, "Total time", _total_time);
      utils::secureLogValue(logger, "\tTrees construction time", _knnStatistics._trees_construction_time, true, 1);
      utils::secureLogValue(logger, "\tAKNN time", _knnStatistics._aknn_time, true, 3);
      utils::secureLogValue(logger, "\tDistributions time", _distribution_time, true, 2);
      utils::secureLog(logger, "--------------------------------------------------------------\n");
    }


    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::HDJointProbabilityGenerator() :
      _logger(nullptr)
    {

    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeJointProbabilityDistribution(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int>      indices;

      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, distribution, params);
      symmetrize(distribution);
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int>      indices;

      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, distribution, params);
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, std::vector<scalar_type>& probabilities, std::vector<int>& indices, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");

      std::vector<scalar_type>  distances_squared;
      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, probabilities, params);
    }

    template <typename scalar, typename sparse_scalar_matrix>

    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeHighDimensionalDistances(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, std::vector<scalar_type>& distances_squared, std::vector<int>& indices, Parameters& params) {

      KnnParameters knnParams;

      knnParams._perplexity = params._perplexity;
      knnParams._perplexity_multiplier = params._perplexity_multiplier;
      knnParams._num_trees = params._num_trees;
      knnParams._num_checks = params._num_checks;
      knnParams._aknn_algorithm = params._aknn_algorithm;
      knnParams._aknn_metric = params._aknn_metric;
      knnParams._aknn_algorithmP1 = params._aknn_algorithmP1;
      knnParams._aknn_algorithmP2 = params._aknn_algorithmP2;

      computeApproximateNearestNeighbors(high_dimensional_data, num_dim, num_dps, knnParams, distances_squared, indices, _statistics._knnStatistics, _logger);
    }


    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int>& indices, int nn, sparse_scalar_matrix& distribution, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const int n = distribution.size();

#ifdef __USE_GCD__
      __block scalar_vector_type temp_vector(distances_squared.size(), 0);
#else
      scalar_vector_type temp_vector(distances_squared.size(), 0);
#endif //__USE_GCD__

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 193.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
        const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
          distances_squared.begin() + j * nn, //check squared
          distances_squared.begin() + (j + 1)*nn,
          temp_vector.begin() + j * nn,
          temp_vector.begin() + (j + 1)*nn,
          params._perplexity,
          200,
          1e-5,
          0
          );
      }
#ifdef __USE_GCD__
      );
#endif

      for (int j = 0; j < n; ++j) {
        for (int k = 1; k < nn; ++k) {
          const unsigned int i = j * nn + k;
          // if items do not have all the same number of neighbors this is indicated by -1
          if (indices[i] == -1)
            continue;
          distribution[j][indices[i]] = temp_vector[i];
        }
      }
      }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int>& indices, sparse_scalar_matrix& distribution, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const int n = distribution.size();

      const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
#ifdef __USE_GCD__
      __block scalar_vector_type temp_vector(distances_squared.size(), 0);
#else
      scalar_vector_type temp_vector(distances_squared.size(), 0);
#endif //__USE_GCD__

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 193.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
        const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
          distances_squared.begin() + j * nn, //check squared
          distances_squared.begin() + (j + 1)*nn,
          temp_vector.begin() + j * nn,
          temp_vector.begin() + (j + 1)*nn,
          params._perplexity,
          200,
          1e-5,
          0
          );
      }
#ifdef __USE_GCD__
      );
#endif

      for (int j = 0; j < n; ++j) {
        for (int k = 1; k < nn; ++k) {
          const unsigned int i = j * nn + k;
          distribution[j][indices[i]] = temp_vector[i];
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int>& indices, std::vector<scalar_type>& probabilities, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");

      const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
      const int n = indices.size() / nn;

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 232.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
        const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
          distances_squared.begin() + j * nn, //check squared
          distances_squared.begin() + (j + 1)*nn,
          probabilities.begin() + j * nn,
          probabilities.begin() + (j + 1)*nn,
          params._perplexity,
          200,
          1e-5,
          0
          );
      }
#ifdef __USE_GCD__
      );
#endif
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::symmetrize(sparse_scalar_matrix& distribution) {
      const int n = distribution.size();
      for (int j = 0; j < n; ++j) {
        for (auto& e : distribution[j]) {
          const unsigned int i = e.first;
          scalar new_val = (distribution[j][i] + distribution[i][j])*0.5;
          distribution[j][i] = new_val;
          distribution[i][j] = new_val;
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributionsFromDistanceMatrix(const std::vector<scalar_type>& squared_distance_matrix, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const int n = num_dps;
      const unsigned int nn = num_dps;
#ifdef __USE_GCD__
      __block scalar_vector_type temp_vector(num_dps*num_dps, 0);
#else
      scalar_vector_type temp_vector(num_dps*num_dps, 0);
#endif //__USE_GCD__
      distribution.clear();
      distribution.resize(n);

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 193.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
        const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
          squared_distance_matrix.begin() + j * nn, //check squared
          squared_distance_matrix.begin() + (j + 1)*nn,
          temp_vector.begin() + j * nn,
          temp_vector.begin() + (j + 1)*nn,
          params._perplexity,
          200,
          1e-5,
          j
          );
      }
#ifdef __USE_GCD__
      );
#endif

      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < nn; ++k) {
          const unsigned int i = j * nn + k;
          distribution[j][k] = temp_vector[i];
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////7


  }
}
#endif 

