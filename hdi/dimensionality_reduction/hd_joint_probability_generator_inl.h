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
#include "hdi/utils/knn_utils.h"
#include <random>
#include <chrono>
#include <unordered_set>
#include <numeric>
#include "omp.h"

#ifdef HNSWLIB_FOUND
#ifdef _MSC_VER
#if (_MSC_VER >= 1910)
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#define HNSWLIB_SUPPORTED
#endif //__cplusplus >=201103
#else // _MSC_VER
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#define HNSWLIB_SUPPORTED
#endif
#endif // HNSWLIB_FOUND

#ifdef __USE_ANNOY__
#include "annoylib.h"
#include "kissrandom.h"
#endif // __USE_ANNOY__


#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif // __USE_GCD__

#pragma warning( push )
#pragma warning( disable : 4267)
#pragma warning( push )
#pragma warning( disable : 4291)
#pragma warning( push )
#pragma warning( disable : 4996)
#pragma warning( push )
#pragma warning( disable : 4018)
#pragma warning( push )
#pragma warning( disable : 4244)
#include "flann/flann.h"
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )

namespace hdi {
  namespace dr {
    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Parameters::Parameters() :
      _perplexity(30),
      _perplexity_multiplier(3),
      _num_trees(4),
      _num_checks(1024),
      _aknn_algorithm(hdi::utils::KNN_FLANN),
      _aknn_metric(hdi::utils::KNN_METRIC_EUCLIDEAN),
      _aknn_algorithmP1(16), // default parameter for HNSW
      _aknn_algorithmP2(200) // default parameter for HNSW
    {}

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::Statistics() :
      _total_time(0),
      _trees_construction_time(0),
      _aknn_time(0),
      _distribution_time(0)
    {}

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::reset() {
      _total_time = 0;
      _trees_construction_time = 0;
      _aknn_time = 0;
      _distribution_time = 0;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::log(utils::AbstractLog* logger)const {
      utils::secureLog(logger, "\n-------- HD Joint Probability Generator Statistics -----------");
      utils::secureLogValue(logger, "Total time", _total_time);
      utils::secureLogValue(logger, "\tTrees construction time", _trees_construction_time, true, 1);
      utils::secureLogValue(logger, "\tAKNN time", _aknn_time, true, 3);
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

      // Fallback to FLANN if others are not supported
#ifndef HNSWLIB_SUPPORTED
      if (params._aknn_algorithm == hdi::utils::KNN_HNSW)
      {
        hdi::utils::secureLog(_logger, "HNSW not available, falling back to FLANN");
        params._aknn_algorithm = hdi::utils::KNN_FLANN;
      }
#endif // HNSWLIB_SUPPORTED

#ifndef __USE_ANNOY__
      if (params._aknn_algorithm == hdi::utils::KNN_ANNOY)
      {
        params._aknn_algorithm = hdi::utils::KNN_FLANN;
      }
#endif // __USE_ANNOY__

      if (params._aknn_algorithm == hdi::utils::KNN_FLANN)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with Flann...");
        flann::Matrix<scalar_type> dataset(high_dimensional_data, num_dps, num_dim);
        flann::Matrix<scalar_type> query(high_dimensional_data, num_dps, num_dim);

        flann::Index<flann::L2<scalar_type> > index(dataset, flann::KDTreeIndexParams(params._num_trees));
        const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
        distances_squared.resize(num_dps*nn);
        indices.resize(num_dps*nn);
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);
          index.buildIndex();
        }
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);
          flann::Matrix<int> indices_mat(indices.data(), query.rows, nn);
          flann::Matrix<scalar_type> dists_mat(distances_squared.data(), query.rows, nn);
          flann::SearchParams flann_params(params._num_checks);
          flann_params.cores = 0; //all cores
          index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
        }
      }
      else if (params._aknn_algorithm == hdi::utils::KNN_HNSW)
      {
#ifdef HNSWLIB_SUPPORTED
        hdi::utils::secureLog(_logger, "Computing approximated knn with HNSWLIB...");

        hnswlib::SpaceInterface<float> *space = NULL;
        switch (params._aknn_metric) {
        case hdi::utils::KNN_METRIC_EUCLIDEAN:
          space = new hnswlib::L2Space(num_dim);
          break;
        case hdi::utils::KNN_METRIC_INNER_PRODUCT:
          space = new hnswlib::InnerProductSpace(num_dim);
          break;
        default:
          space = new hnswlib::L2Space(num_dim);
          break;
        }

        hnswlib::HierarchicalNSW<scalar> appr_alg(space, num_dps, params._aknn_algorithmP1, params._aknn_algorithmP2, 0);
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);
          appr_alg.addPoint((void*)high_dimensional_data, (std::size_t) 0);
#pragma omp parallel for
          for (int i = 1; i < num_dps; ++i)
          {
            appr_alg.addPoint((void*)(high_dimensional_data + (i*num_dim)), (hnswlib::labeltype) i);
          }
        }
        const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
        distances_squared.resize(num_dps*nn);
        indices.resize(num_dps*nn);
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);
#pragma omp parallel for
          for (int i = 0; i < num_dps; ++i)
          {
            auto top_candidates = appr_alg.searchKnn(high_dimensional_data + (i*num_dim), (hnswlib::labeltype)nn);
            while (top_candidates.size() > nn) {
              top_candidates.pop();
            }
            auto *distances_offset = distances_squared.data() + (i*nn);
            auto indices_offset = indices.data() + (i*nn);
            int j = 0;
            while (top_candidates.size() > 0) {
              auto rez = top_candidates.top();
              distances_offset[nn - j - 1] = rez.first;
              indices_offset[nn - j - 1] = appr_alg.getExternalLabel(rez.second);
              top_candidates.pop();
              ++j;
            }
          }
        }
#endif // HNSWLIB_SUPPORTED
      }
      else if (params._aknn_algorithm == hdi::utils::KNN_ANNOY)
      {
#ifdef __USE_ANNOY__
        int k = (int)params._perplexity * params._perplexity_multiplier + 1;
        int search_k = k * params._num_trees;

        distances_squared.resize(num_dps * k);
        indices.resize(num_dps * k);

        AnnoyIndexInterface<int, double>* tree = NULL;
        switch (params._aknn_metric) {
        case hdi::utils::KNN_METRIC_EUCLIDEAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int, double, Euclidean, Kiss32Random>(num_dim);
          break;
        case hdi::utils::KNN_METRIC_COSINE:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Cosine distances ...");
          tree = new AnnoyIndex<int, double, Angular, Kiss32Random>(num_dim);
          break;
        case hdi::utils::KNN_METRIC_MANHATTAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          tree = new AnnoyIndex<int, double, Manhattan, Kiss32Random>(num_dim);
          break;
          //case hdi::utils::KNN_METRIC_HAMMING:
          //  hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          //  tree = new AnnoyIndex<int, double, Hamming, Kiss32Random>(num_dim);
          //  break;
        case hdi::utils::KNN_METRIC_DOT:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Dot product distances ...");
          tree = new AnnoyIndex<int, double, DotProduct, Kiss32Random>(num_dim);
          break;
        default:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int, double, Euclidean, Kiss32Random>(num_dim);
          break;
        }

        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);

          for (int i = 0; i < num_dps; ++i) {
            double* vec = new double[num_dim];
            for (int z = 0; z < num_dim; ++z) {
              vec[z] = high_dimensional_data[i * num_dim + z];
            }
            tree->add_item(i, vec);
          }
          tree->build(params._num_trees);

          // Sample check if it returns enough neighbors
          std::vector<int> closest;
          std::vector<double> closest_distances;
          for (int n = 0; n < 100; n++) {
            tree->get_nns_by_item(n, k, search_k, &closest, &closest_distances);
            unsigned int neighbors_count = closest.size();
            if (neighbors_count < k) {
              printf("Requesting %d neighbors, but ANNOY returned only %u. Please increase search_k\n", k, neighbors_count);
              return;
            }
          }
        }

        hdi::utils::secureLog(_logger, "Done building tree. Beginning nearest neighbor search... ");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);

#pragma omp parallel for
          for (int n = 0; n < num_dps; n++)
          {
            // Find nearest neighbors
            std::vector<int> closest;
            std::vector<double> closest_distances;
            tree->get_nns_by_item(n, k, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned int m = 0; m < k; m++) {
              indices[n * k + m] = closest[m];
              distances_squared[n * k + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
        delete tree;
#endif // __USE_ANNOY__
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

