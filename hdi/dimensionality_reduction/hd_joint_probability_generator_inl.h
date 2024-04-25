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

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"

#pragma warning( push )
#pragma warning( disable : 4477) // mismatch in type for fprintf with (unused) verbose setting
#include "annoylib.h"
#pragma warning( pop )
#include "kissrandom.h"

#include "flann/flann.h"

#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif

namespace hdi {
  namespace dr {
    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Parameters::Parameters() :
      _perplexity(30),
      _perplexity_multiplier(3),
      _num_trees(4),
      _num_checks(1024),
      _aknn_algorithm(hdi::dr::KNN_FLANN),
      _aknn_metric(hdi::dr::KNN_METRIC_EUCLIDEAN),
      _aknn_algorithmP1(16), // default parameter for HNSW
      _aknn_algorithmP2(200) // default parameter for HNSW
    {}

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::Statistics() :
      _total_time(0),
      _trees_construction_time(0),
      _aknn_time(0),
      _distribution_time(0)
    {}

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::reset() {
      _total_time = 0;
      _trees_construction_time = 0;
      _aknn_time = 0;
      _distribution_time = 0;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::log(utils::AbstractLog* logger)const {
      utils::secureLog(logger, "\n-------- HD Joint Probability Generator Statistics -----------");
      utils::secureLogValue(logger, "Total time", _total_time);
      utils::secureLogValue(logger, "\tTrees construction time", _trees_construction_time, true, 1);
      utils::secureLogValue(logger, "\tAKNN time", _aknn_time, true, 3);
      utils::secureLogValue(logger, "\tDistributions time", _distribution_time, true, 2);
      utils::secureLog(logger, "--------------------------------------------------------------\n");
    }


    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::HDJointProbabilityGenerator() :
      _logger(nullptr)
    {

    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeJointProbabilityDistribution(scalar_type* high_dimensional_data, unsigned_int_type num_dim, unsigned_int_type num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int_type>      indices;

      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, distribution, params);
      symmetrize(distribution);
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned_int_type num_dim, unsigned_int_type num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int_type>      indices;

      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, distribution, params);
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned_int_type num_dim, unsigned_int_type num_dps, std::vector<scalar_type>& probabilities, std::vector<int_type>& indices, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger, "Computing the HD joint probability distribution...");

      std::vector<scalar_type>  distances_squared;
      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared, indices, probabilities, params);
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeHighDimensionalDistances(scalar_type* high_dimensional_data, unsigned_int_type num_dim, unsigned_int_type num_dps, std::vector<scalar_type>& distances_squared, std::vector<int_type>& indices, Parameters& params) {

      const unsigned_int_type nn = params._perplexity * params._perplexity_multiplier + 1;
      distances_squared.resize(num_dps * nn);
      indices.resize(num_dps * nn);

      if (params._aknn_algorithm == hdi::dr::KNN_FLANN)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with Flann...");
        flann::Matrix<scalar_type> dataset(high_dimensional_data, num_dps, num_dim);
        flann::Matrix<scalar_type> query(high_dimensional_data, num_dps, num_dim);

        flann::Index<flann::L2<scalar_type> > index(dataset, flann::KDTreeIndexParams(params._num_trees));

        utils::secureLog(_logger, "\tBuilding the trees...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);
          index.buildIndex();
        }

        utils::secureLog(_logger, "\tAKNN queries...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);

          flann::Matrix<scalar_type> dists_mat(distances_squared.data(), query.rows, nn);
          flann::SearchParams flann_params(params._num_checks);
          flann_params.cores = 0; //all cores

          // flann's knnSearch only accepts int and size_t indices
          if constexpr (std::is_same_v<int_type, size_t> || std::is_same_v<int_type, int>)
          {
            flann::Matrix<int_type> indices_mat(indices.data(), query.rows, nn);
            index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
          }
          else
          {
            std::vector<size_t> flannIndices;
            flannIndices.resize(indices.size());
            flann::Matrix<size_t> indices_mat(flannIndices.data(), query.rows, nn);
            index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
#pragma omp parallel for
            for (std::int64_t i = 0; i < indices.size(); ++i)
              indices[i] = static_cast<int_type>(flannIndices[i]);
          }
        }
      }
      else if (params._aknn_algorithm == hdi::dr::KNN_HNSW)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with HNSW Lib...");

        hnswlib::SpaceInterface<scalar> *space = NULL;
        switch (params._aknn_metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          space = new hnswlib::L2Space(num_dim);
          break;
        case hdi::dr::KNN_METRIC_INNER_PRODUCT:
          space = new hnswlib::InnerProductSpace(num_dim);
          break;
        default:
          space = new hnswlib::L2Space(num_dim);
          break;
        }

        utils::secureLog(_logger, "\tBuilding the search structure...");
        hnswlib::HierarchicalNSW<scalar> appr_alg(space, num_dps, params._aknn_algorithmP1, params._aknn_algorithmP2, 0);
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);
          appr_alg.addPoint((void*)high_dimensional_data, (std::size_t) 0);
          unsigned num_threads = std::thread::hardware_concurrency();
          hnswlib::ParallelFor(1, num_dps, num_threads, [&](size_t i, size_t threadId) {
            appr_alg.addPoint((void*)(high_dimensional_data + (i*num_dim)), (hnswlib::labeltype) i);
          });
        }

        utils::secureLog(_logger, "\tAKNN queries...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);
#pragma omp parallel for
          for (std::int64_t i = 0; i < num_dps; ++i)
          {
            auto top_candidates = appr_alg.searchKnn(high_dimensional_data + (i*num_dim), (hnswlib::labeltype)nn);
            while (top_candidates.size() > nn) {
              top_candidates.pop();
            }
            auto *distances_offset = distances_squared.data() + (i*nn);
            auto indices_offset = indices.data() + (i*nn);
            std::int64_t j = 0;
            assert(top_candidates.size() == nn);
            while (top_candidates.size() > 0) {
              auto rez = top_candidates.top();
              distances_offset[nn - j - 1] = rez.first;
              indices_offset[nn - j - 1] = rez.second;
              top_candidates.pop();
              ++j;
            }
          }
        }
      }
      else if (params._aknn_algorithm == hdi::dr::KNN_ANNOY)
      {
        using namespace Annoy;
        utils::secureLog(_logger, "Computing approximated knn with Annoy...");

        int search_k = nn * params._num_trees;

        AnnoyIndexInterface<int_type, scalar>* tree = nullptr;
        switch (params._aknn_metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int_type, scalar, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_COSINE:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Cosine distances ...");
          tree = new AnnoyIndex<int_type, scalar, Angular, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_MANHATTAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          tree = new AnnoyIndex<int_type, scalar, Manhattan, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
          //case hdi::dr::KNN_METRIC_HAMMING:
          //  hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          //  tree = new AnnoyIndex<int_type, scalar, Hamming, Kiss64Random>(num_dim);
          //  break;
        case hdi::dr::KNN_METRIC_DOT:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Dot product distances ...");
          tree = new AnnoyIndex<int_type, scalar, DotProduct, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        default:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int_type, scalar, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        }

        utils::secureLog(_logger, "\tBuilding the search structure...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);

          for (unsigned_int_type i = 0; i < num_dps; ++i) {
            scalar* vec = new scalar[num_dim];
            for (unsigned_int_type z = 0; z < num_dim; ++z) {
              vec[z] = high_dimensional_data[i * num_dim + z];
            }
            tree->add_item(i, vec);
          }
          tree->build(params._num_trees);

          // Sample check if it returns enough neighbors
          std::vector<int_type> closest;
          std::vector<scalar> closest_distances;
          for (int n = 0; n < 100; n++) {
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);
            size_t neighbors_count = closest.size();
            if (neighbors_count < nn) {
              printf("Requesting %d neighbors, but ANNOY returned only %u. Please increase search_k\n", static_cast<int>(nn), static_cast<unsigned int>(neighbors_count));
              return;
            }
          }
        }

        hdi::utils::secureLog(_logger, "Done building tree. Beginning nearest neighbor search... ");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);

#pragma omp parallel for
          for (std::int64_t n = 0; n < num_dps; n++)
          {
            // Find nearest neighbors
            std::vector<int_type> closest;
            std::vector<scalar> closest_distances;
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned_int_type m = 0; m < nn; m++) {
              indices[n * nn + m] = closest[m];
              distances_squared[n * nn + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
        delete tree;
      }
    }


    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int_type>& indices, int nn, sparse_scalar_matrix& distribution, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const size_t n = distribution.size();

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
      for (std::int64_t j = 0; j < n; ++j) {
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

      for (size_t j = 0; j < n; ++j) {
        for (int k = 1; k < nn; ++k) {
          const size_t i = j * nn + k;
          // if items do not have all the same number of neighbors this is indicated by -1
          if (indices[i] == -1)
            continue;
          distribution[j][indices[i]] = temp_vector[i];
        }
      }
      }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int_type>& indices, sparse_scalar_matrix& distribution, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const size_t       n  = distribution.size();
      const unsigned int nn = params._perplexity * params._perplexity_multiplier + 1;

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
      for (std::int64_t j = 0; j < n; ++j) {
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

      for (size_t j = 0; j < n; ++j) {
        for (unsigned int k = 1; k < nn; ++k) {
          const size_t i = j * nn + k;
          distribution[j][indices[i]] = temp_vector[i];
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int_type>& indices, std::vector<scalar_type>& probabilities, Parameters& params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");

      const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
      const size_t n = indices.size() / nn;

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 232.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (std::int64_t j = 0; j < n; ++j) {
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

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::symmetrize(sparse_scalar_matrix& distribution) {
      const size_t n = distribution.size();
      for (size_t j = 0; j < n; ++j) {
        for (auto& e : distribution[j]) {
          const auto i = e.first;
          scalar new_val = (distribution[j][i] + distribution[i][j])*0.5;
          distribution[j][i] = new_val;
          distribution[i][j] = new_val;
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeProbabilityDistributionsFromDistanceMatrix(const std::vector<scalar_type>& squared_distance_matrix, unsigned_int_type num_dps, sparse_scalar_matrix& distribution, Parameters params) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger, "Computing joint-probability distribution...");
      const unsigned_int_type n = num_dps;
      const unsigned_int_type nn = num_dps;
#ifdef __USE_GCD__
      __block scalar_vector_type temp_vector(num_dps*num_dps, 0);
#else
      scalar_vector_type temp_vector(num_dps * num_dps, 0);
#endif //__USE_GCD__
      distribution.clear();
      distribution.resize(n);

#ifdef __USE_GCD__
      std::cout << "GCD dispatch, hd_joint_probability_generator 193.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
      for (std::int64_t j = 0; j < n; ++j) {
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

      for (size_t j = 0; j < n; ++j) {
        for (int k = 0; k < nn; ++k) {
          const size_t i = j * nn + k;
          distribution[j][k] = temp_vector[i];
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////7


  }
}
#endif 

