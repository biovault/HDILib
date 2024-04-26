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


#ifndef HIERARCHICAL_SNE_INL
#define HIERARCHICAL_SNE_INL

#include "hdi/data/map_helpers.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/hierarchical_sne.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/log_progress.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/memory_utils.h"

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"

#pragma warning( push )
#pragma warning( disable : 4477) // mismatch in type for fprintf with (unused) verbose setting
#include "annoylib.h"
#pragma warning( pop )
#include "kissrandom.h"

#include "flann/flann.h"

#include <chrono>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <string>

namespace hdi {
  namespace dr {
    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Parameters::Parameters() :
      _seed(-1),
      _num_neighbors(30),
      _aknn_num_trees(4),
      _aknn_num_checks(1024),
      _aknn_algorithm(hdi::dr::KNN_FLANN),
      _aknn_metric(hdi::dr::KNN_METRIC_EUCLIDEAN),
      _aknn_algorithmP1(16), // default parameter for HNSW
      _aknn_algorithmP2(200), // default parameter for HNSW
      _monte_carlo_sampling(true),
      _mcmcs_num_walks(10),
      _mcmcs_landmark_thresh(1.5),
      _mcmcs_walk_length(10),
      _hard_cut_off(false),
      _hard_cut_off_percentage(0.1f),
      _rs_reduction_factor_per_layer{ static_cast<scalar_type>(.1) },
      _rs_outliers_removal_jumps(10),
      _num_walks_per_landmark(100),
      _transition_matrix_prune_thresh(1.5),
      _out_of_core_computation(false)
    {}

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::Statistics() :
      _total_time(-1),
      _init_knn_time(-1),
      _init_probabilities_time(-1),
      _init_fmc_time(-1),
      _mcmc_sampling_time(-1),
      _landmarks_selection_time(-1),
      _landmarks_selection_num_walks(-1),
      _aoi_time(-1),
      _fmc_time(-1),
      _aoi_num_walks(-1),
      _aoi_sparsity(-1),
      _fmc_sparsity(-1),
      _fmc_effective_sparsity(-1)
    {}
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::reset() {
      _total_time = -1;
      _init_knn_time = -1;
      _init_probabilities_time = -1;
      _init_fmc_time = -1;
      _mcmc_sampling_time = -1;
      _landmarks_selection_time = -1;
      _landmarks_selection_num_walks = -1;
      _aoi_time = -1;
      _fmc_time = -1;
      _aoi_num_walks = -1;
      _aoi_sparsity = -1;
      _fmc_sparsity = -1;
      _fmc_effective_sparsity = -1;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Statistics::log(utils::AbstractLog* logger)const {
      utils::secureLog(logger, "\n--------------- Hierarchical-SNE Statistics ------------------");
      utils::secureLogValue(logger, "Total time", _total_time);
      if (_init_knn_time != -1) { utils::secureLogValue(logger, "\tAKNN graph computation time", _init_knn_time, true, 2); }
      if (_init_probabilities_time != -1) { utils::secureLogValue(logger, "\tTransition probabilities computation time", _init_probabilities_time, true, 1); }
      if (_init_fmc_time != -1) { utils::secureLogValue(logger, "\tFMC computation time", _init_fmc_time, true, 3); }
      if (_mcmc_sampling_time != -1) { utils::secureLogValue(logger, "\tMarkov Chain Monte Carlo sampling time", _mcmc_sampling_time, true, 1); }
      if (_landmarks_selection_time != -1) { utils::secureLogValue(logger, "\tLandmark selection time", _landmarks_selection_time, true, 2); }
      if (_landmarks_selection_num_walks != -1) { utils::secureLogValue(logger, "\tLndks Slct #walks", _landmarks_selection_num_walks, true, 3); }
      if (_aoi_time != -1) { utils::secureLogValue(logger, "\tArea of Influence computation time", _aoi_time, true, 1); }
      if (_fmc_time != -1) { utils::secureLogValue(logger, "\tFMC computation time", _fmc_time, true, 3); }
      if (_aoi_num_walks != -1) { utils::secureLogValue(logger, "\tAoI #walks", _aoi_num_walks, true, 4); }
      if (_aoi_sparsity != -1) { utils::secureLogValue(logger, "\tIs sparsity (%)", _aoi_sparsity * 100, true, 3); }
      if (_fmc_sparsity != -1) { utils::secureLogValue(logger, "\tTs sparsity (%)", _fmc_sparsity * 100, true, 3); }
      if (_fmc_effective_sparsity != -1) { utils::secureLogValue(logger, "\tTs effective sparsity (%)", _fmc_effective_sparsity * 100, true, 2); }
      utils::secureLog(logger, "--------------------------------------------------------------\n");
    }

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    scalar HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::Scale::mimMemoryOccupation()const {
      scalar_type mem(0);

      mem += _landmark_to_original_data_idx.capacity() * sizeof(unsigned_int_type);
      mem += _landmark_to_previous_scale_idx.capacity() * sizeof(unsigned_int_type);
      mem += _landmark_weight.capacity() * sizeof(scalar_type);
      for (int i = 0; i < _transition_matrix.size(); ++i) {
        mem += _transition_matrix[i].size()*(sizeof(unsigned_int_type) + sizeof(scalar_type));
      }

      mem += _previous_scale_to_landmark_idx.capacity() * sizeof(int_type);
      for (int i = 0; i < _area_of_influence.size(); ++i) {
        mem += _area_of_influence[i].size()*(sizeof(unsigned_int_type) + sizeof(scalar_type));
      }

      return mem / 1024 / 1024;
    }

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::HierarchicalSNE() :
      _hierarchy(),
      _landmarks_to_datapoints(),
      _dimensionality(0),
      _num_dps(0),
      _high_dimensional_data(nullptr),
      _initialized(false),
      _verbose(false),
      _params(),
      _logger(nullptr),
      _statistics()
    {

    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::reset() {
      _initialized = false;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::clear() {
      _high_dimensional_data = nullptr;
      _initialized = false;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getHighDimensionalDescriptor(scalar_vector_type& data_point, map_key_type handle)const {
      data_point.resize(_dimensionality);
      for (unsigned_int_type i = 0; i < _dimensionality; ++i) {
        data_point[i] = *(_high_dimensional_data + handle * _dimensionality + i);
      }
    }

    /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::initialize(scalar_type* high_dimensional_data, unsigned_int_type num_dps, Parameters params) {
      _statistics.reset();
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);
      utils::secureLog(_logger, "Initializing Hierarchical-SNE...");
      _params = params;
      _high_dimensional_data = high_dimensional_data;
      _num_dps = num_dps;
      _landmarks_to_datapoints.clear();

      utils::secureLogValue(_logger, "Number of data points", _num_dps);
      initializeFirstScale();

      _initialized = true;
      utils::secureLog(_logger, "Initialization complete!");
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::initialize(const sparse_scalar_matrix_type& similarities, Parameters params) {
      _statistics.reset();
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);
      utils::secureLog(_logger, "Initializing Hierarchical-SNE...");
      _params = params;
      _high_dimensional_data = nullptr;
      _num_dps = similarities.size();
      _landmarks_to_datapoints.clear();

      utils::secureLogValue(_logger, "Number of data points", _num_dps);
      initializeFirstScale(similarities);

      _initialized = true;
      utils::secureLog(_logger, "Initialization complete!");
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::addScale() {
      _statistics.reset();
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);
      bool res(true);
      if (_params._out_of_core_computation) {
        addScaleOutOfCoreImpl();
      }
      else {
        addScaleImpl();
      }
      _statistics.log(_logger);
      return res;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::computeNeighborhoodGraph(scalar_vector_type& distance_based_probabilities, std::vector<int>& neighborhood_graph) {

      const unsigned_int_type nn = _params._num_neighbors + 1;
      scalar_type perplexity = _params._num_neighbors / 3.;
      neighborhood_graph.resize(_num_dps*nn);
      distance_based_probabilities.resize(_num_dps*nn);
        
      if(_params._aknn_algorithm == hdi::dr::KNN_FLANN)
      {
        utils::secureLog(_logger, "Computing approximated neighborhood graph with Flann...");
        flann::Matrix<scalar_type> dataset(_high_dimensional_data, _num_dps, _dimensionality);
        flann::Matrix<scalar_type> query(_high_dimensional_data, _num_dps, _dimensionality);

        flann::Index<flann::L2<scalar_type> > index(dataset, flann::KDTreeIndexParams(_params._aknn_num_trees));

        utils::secureLog(_logger, "\tBuilding the trees...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_knn_time);
          index.buildIndex();
        }

        utils::secureLog(_logger, "\tAKNN queries...");
        {
          flann::Matrix<scalar_type> dists_mat(distance_based_probabilities.data(), query.rows, nn);
          flann::SearchParams flann_params(_params._aknn_num_checks);
          flann_params.cores = 0; //all cores

          // flann's knnSearch only accepts int and size_t indices
          if constexpr (std::is_same_v<int_type, size_t> || std::is_same_v<int_type, int>)
          {
            flann::Matrix<int_type> indices_mat(neighborhood_graph.data(), query.rows, nn);
            index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
          }
          else
          {
            std::vector<size_t> flannIndices;
            flannIndices.resize(neighborhood_graph.size());
            flann::Matrix<size_t> indices_mat(flannIndices.data(), query.rows, nn);
            index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
#pragma omp parallel for
            for (std::int64_t i = 0; i < neighborhood_graph.size(); ++i)
              neighborhood_graph[i] = static_cast<int_type>(flannIndices[i]);
          }
        }
      }
      else if (_params._aknn_algorithm == hdi::dr::KNN_HNSW)
      {
        utils::secureLog(_logger, "Computing approximated neighborhood graph with HNSW Lib...");

        hnswlib::SpaceInterface<float> *space = NULL;
        switch (_params._aknn_metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          space = new hnswlib::L2Space(_dimensionality);
          break;
        case hdi::dr::KNN_METRIC_INNER_PRODUCT:
          space = new hnswlib::InnerProductSpace(_dimensionality);
          break;
        default:
          space = new hnswlib::L2Space(_dimensionality);
          break;
        }
        hnswlib::HierarchicalNSW<scalar_type> appr_alg(space, _num_dps, _params._aknn_algorithmP1, _params._aknn_algorithmP2, 0);

        utils::secureLog(_logger, "\tBuilding the search structure...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_knn_time);
          appr_alg.addPoint((void*)_high_dimensional_data, (std::size_t)0);
          unsigned num_threads = std::thread::hardware_concurrency();
          hnswlib::ParallelFor(1, _num_dps, num_threads, [&](size_t i, size_t threadId) {
            appr_alg.addPoint((void*)(_high_dimensional_data + (i * _dimensionality)), (hnswlib::labeltype)i);
            });
        }

        utils::secureLog(_logger, "\tAKNN queries...");
#pragma omp parallel for
        for (std::int64_t i = 0; i < _num_dps; ++i)
        {
          auto top_candidates = appr_alg.searchKnn(_high_dimensional_data + (i*_dimensionality), (hnswlib::labeltype)nn);
          while (top_candidates.size() > nn) {
            top_candidates.pop();
          }
          auto *distances = distance_based_probabilities.data() + (i*nn);
          auto *indices = neighborhood_graph.data() + (i*nn);
          std::int64_t j = 0;
          assert(top_candidates.size() == nn);
          while (top_candidates.size() > 0)
          {
            auto rez = top_candidates.top();
            distances[nn - j - 1] = rez.first;
            indices[nn - j - 1] = rez.second;
            top_candidates.pop();
            ++j;
          }
        }
      }
      else if (_params._aknn_algorithm == hdi::dr::KNN_ANNOY)
      {
        using namespace Annoy;
        hdi::utils::secureLog(_logger, "Computing approximated neighborhood graph with Annoy...");

        int search_k = nn * _params._aknn_num_trees;

        AnnoyIndexInterface<int_type, scalar>* tree = nullptr;
        switch (_params._aknn_metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int_type, scalar, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(_dimensionality);
          break;
        case hdi::dr::KNN_METRIC_COSINE:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Cosine distances ...");
          tree = new AnnoyIndex<int_type, scalar, Angular, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(_dimensionality);
          break;
        case hdi::dr::KNN_METRIC_MANHATTAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          tree = new AnnoyIndex<int_type, scalar, Manhattan, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(_dimensionality);
          break;
          //case hdi::dr::KNN_METRIC_HAMMING:
          //  hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          //  tree = new AnnoyIndex<int_type, scalar, Hamming, Kiss64Random>(num_dim);
          //  break;
        case hdi::dr::KNN_METRIC_DOT:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Dot product distances ...");
          tree = new AnnoyIndex<int_type, scalar, DotProduct, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(_dimensionality);
          break;
        default:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int_type, scalar, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(_dimensionality);
          break;
        }

        utils::secureLog(_logger, "\tBuilding the search structure...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_knn_time);

          for (unsigned_int_type i = 0; i < _num_dps; ++i) {
            scalar* vec = new scalar[_dimensionality];
            for (unsigned_int_type z = 0; z < _dimensionality; ++z) {
              vec[z] = _high_dimensional_data[i * _dimensionality + z];
            }
            tree->add_item(i, vec);
          }
          tree->build(_params._aknn_num_trees);

          // Sample check if it returns enough neighbors
          std::vector<int_type> closest;
          std::vector<scalar> closest_distances;
          for (size_t n = 0; n < 100; n++) {
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);
            size_t neighbors_count = closest.size();
            if (neighbors_count < nn) {
              printf("Requesting %d neighbors, but ANNOY returned only %u. Please increase search_k\n", static_cast<int>(nn), static_cast<unsigned int>(neighbors_count));
              return;
            }
          }

          hdi::utils::secureLog(_logger, "Done building trees. Beginning nearest neighbor search... ");

#pragma omp parallel for
          for (int_type n = 0; n < _num_dps; n++)
          {
            // Find nearest neighbors
            std::vector<int_type> closest;
            std::vector<scalar> closest_distances;
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned_int_type m = 0; m < nn; m++) {
              neighborhood_graph[n * nn + m] = closest[m];
              distance_based_probabilities[n * nn + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
        delete tree;
      }

      {
        utils::secureLog(_logger, "\tFMC computation...");
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_probabilities_time);
        
        //#ifdef __USE_GCD__
        //        std::cout << "GCD dispatch, hierarchical_sne_inl 253.\n";
        //        dispatch_apply(_num_dps, dispatch_get_global_queue(0, 0), ^(size_t d) {
        //#else
#pragma omp parallel for
        for (int_type d = 0; d < _num_dps; ++d) {
          //#endif //__USE_GCD__
                    //It could be that the point itself is not the nearest one if two points are identical... I want the point itself to be the first one!
          if (neighborhood_graph[d*nn] != d) {
            unsigned_int_type to_swap = d * nn;
            for (; to_swap < d*nn + (nn-1); ++to_swap) {
              if (neighborhood_graph[to_swap] == d)
                break;
            }
            std::swap(neighborhood_graph[nn*d], neighborhood_graph[to_swap]);
            std::swap(distance_based_probabilities[nn*d], distance_based_probabilities[to_swap]);
          }
          scalar_vector_type temp_probability(nn, 0);
          utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
            distance_based_probabilities.cbegin() + d * nn,
            distance_based_probabilities.cbegin() + (d + 1)*nn,
            temp_probability.begin(),
            temp_probability.begin() + nn,
            perplexity,
            200,
            1e-5,
            0
            );
          distance_based_probabilities[d*nn] = 0;
          for (unsigned_int_type n = 1; n < nn; ++n) {
            distance_based_probabilities[d*nn + n] = temp_probability[n];
          }
        }
        //#ifdef __USE_GCD__
        //        );
        //#endif
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::initializeFirstScale() {
      utils::secureLog(_logger, "Initializing the first scale...");

      _hierarchy.clear();
      _hierarchy.push_back(Scale());
      Scale& scale = _hierarchy[0];

      scalar_vector_type distance_based_probabilities;
      std::vector<int> neighborhood_graph;

      computeNeighborhoodGraph(distance_based_probabilities, neighborhood_graph);
      unsigned_int_type nn = _params._num_neighbors + 1;

      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_fmc_time);
        utils::secureLog(_logger, "Creating transition matrix...");
        scale._landmark_to_original_data_idx.resize(_num_dps);
        std::iota(scale._landmark_to_original_data_idx.begin(), scale._landmark_to_original_data_idx.end(), 0);
        scale._landmark_to_previous_scale_idx = scale._landmark_to_original_data_idx;
        scale._landmark_weight.resize(_num_dps, 1);
        scale._transition_matrix.resize(_num_dps);

        //#ifdef __USE_GCD__
        //        std::cout << "GCD dispatch, hierarchical_sne_inl 253.\n";
        //        dispatch_apply(_num_dps, dispatch_get_global_queue(0, 0), ^(size_t i) {
        //#else
#pragma omp parallel for
        for (int_type i = 0; i < _num_dps; ++i) {
          //#endif //__USE_GCD__
          scalar_type sum = 0;
          for (unsigned_int_type n = 1; n < nn; ++n) {
            unsigned_int_type idx = i * nn + n;
            auto v = distance_based_probabilities[idx];
            sum += v;
            scale._transition_matrix[i][neighborhood_graph[idx]] = v;
          }
        }
        //#ifdef __USE_GCD__
        //        );
        //#endif

      }

      utils::secureLogValue(_logger, "Min memory requirements (MB)", scale.mimMemoryOccupation());
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::initializeFirstScale(const sparse_scalar_matrix_type& similarities) {
      utils::secureLog(_logger, "Initializing the first scale...");

      _hierarchy.clear();
      _hierarchy.push_back(Scale());
      Scale& scale = _hierarchy[0];

      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._init_fmc_time);
        utils::secureLog(_logger, "Creating transition matrix...");
        scale._landmark_to_original_data_idx.resize(_num_dps);
        std::iota(scale._landmark_to_original_data_idx.begin(), scale._landmark_to_original_data_idx.end(), 0);
        scale._landmark_to_previous_scale_idx = scale._landmark_to_original_data_idx;
        scale._landmark_weight.resize(_num_dps, 1);
        scale._transition_matrix = similarities;
      }

      utils::secureLogValue(_logger, "Min memory requirements (MB)", scale.mimMemoryOccupation());
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::selectLandmarks(const Scale& previous_scale, Scale& scale, unsigned_int_type& selected_landmarks) {
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._landmarks_selection_time);
      utils::secureLog(_logger, "Landmark selection with fixed reduction...");
      const unsigned_int_type previous_scale_dp = previous_scale._transition_matrix.size();
      const unsigned_int_type num_landmarks = previous_scale_dp * _params._rs_reduction_factor_per_layer;

      std::default_random_engine generator(seed());
      std::uniform_int_distribution<> distribution_int(0, previous_scale_dp - 1);
      std::uniform_real_distribution<double> distribution_real(0.0, 1.0);

      scale._landmark_to_original_data_idx.resize(num_landmarks, 0);
      scale._landmark_to_previous_scale_idx.resize(num_landmarks, 0);
      scale._landmark_weight.resize(num_landmarks, 0);
      scale._previous_scale_to_landmark_idx.resize(previous_scale_dp, -1);
      scale._transition_matrix.resize(num_landmarks);
      scale._area_of_influence.resize(previous_scale_dp);

      int num_tries = 0;
      selected_landmarks = 0;
      while (selected_landmarks < num_landmarks) {
        ++num_tries;
        int idx = distribution_int(generator);
        assert(idx >= 0);
        assert(idx < _num_dps);

        if (_params._rs_outliers_removal_jumps > 0) {
          idx = randomWalk(idx, _params._rs_outliers_removal_jumps, previous_scale._transition_matrix, distribution_real, generator);
        }

        if (scale._previous_scale_to_landmark_idx[idx] != -1) {
          continue;
        }

        scale._previous_scale_to_landmark_idx[idx] = selected_landmarks;
        scale._landmark_to_original_data_idx[selected_landmarks] = previous_scale._landmark_to_original_data_idx[idx];
        scale._landmark_to_previous_scale_idx[selected_landmarks] = idx;

        ++selected_landmarks;
      }
      _statistics._landmarks_selection_num_walks = num_tries * _params._rs_outliers_removal_jumps;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::selectLandmarksWithStationaryDistribution(const Scale& previous_scale, Scale& scale, unsigned_int_type& selected_landmarks) {
      utils::secureLog(_logger, "Landmark selection...");
      const unsigned_int_type previous_scale_dp = previous_scale._transition_matrix.size();
      unsigned_int_type count = 0;
      unsigned_int_type thresh = _params._mcmcs_num_walks * _params._mcmcs_landmark_thresh;
      //__block std::vector<unsigned_int_type> importance_sampling(previous_scale_dp,0);
      std::vector<unsigned_int_type> importance_sampling(previous_scale_dp, 0);

      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._mcmc_sampling_time);

        //__block std::default_random_engine generator(seed());
        //__block std::uniform_real_distribution<double> distribution_real(0.0, 1.0);
        std::default_random_engine generator(seed());
        std::uniform_real_distribution<double> distribution_real(0.0, 1.0);
        selected_landmarks = 0;

        utils::secureLog(_logger, "Monte Carlo Approximation...");
        unsigned_int_type invalid = std::numeric_limits<unsigned_int_type>::max();

        //#ifdef __USE_GCD__
        //        std::cout << "GCD dispatch, hierarchical_sne_inl 391.\n";
        //        dispatch_apply(previous_scale_dp, dispatch_get_global_queue(0, 0), ^(size_t d) {
        //#else
#pragma omp parallel for
        for (int_type d = 0; d < previous_scale_dp; ++d) {
          //#endif //__USE_GCD__
          for (unsigned_int_type p = 0; p < _params._mcmcs_num_walks; ++p) {
            int_type idx = d;
            idx = randomWalk(idx, _params._mcmcs_walk_length, previous_scale._transition_matrix, distribution_real, generator);
            if (idx != invalid) {
              ++importance_sampling[idx];
            }
          }
        }
        //#ifdef __USE_GCD__
        //        );
        //#endif

                // cheap hack to get the hard cutoff in, still computes the data driven part which should probably be replaced...
        if (_params._hard_cut_off)
        {
          std::vector<unsigned_int_type> importance_sampling_sort = importance_sampling;
          std::sort(importance_sampling_sort.begin(), importance_sampling_sort.end());
          unsigned_int_type cutoff = importance_sampling_sort[(importance_sampling_sort.size() - 1) * (1.0f - _params._hard_cut_off_percentage)];
          thresh = cutoff;
        }

        _statistics._landmarks_selection_num_walks = previous_scale_dp * _params._mcmcs_num_walks;

        for (unsigned_int_type i = 0; i < previous_scale_dp; ++i) {
          if (importance_sampling[i] > thresh)
            ++count;
        }
      }

      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._landmarks_selection_time);
        utils::secureLog(_logger, "Selection...");

        scale._previous_scale_to_landmark_idx.resize(previous_scale_dp, -1);
        scale._area_of_influence.resize(previous_scale_dp);
        scale._landmark_to_original_data_idx.resize(count);
        scale._landmark_to_previous_scale_idx.resize(count);
        scale._landmark_weight.resize(count);
        scale._transition_matrix.resize(count);
        selected_landmarks = 0;

        for (unsigned_int_type i = 0; i < previous_scale_dp; ++i) {
          if (importance_sampling[i] > thresh) {
            scale._previous_scale_to_landmark_idx[i] = selected_landmarks;
            scale._landmark_to_original_data_idx[selected_landmarks] = previous_scale._landmark_to_original_data_idx[i];
            scale._landmark_to_previous_scale_idx[selected_landmarks] = i;
            ++selected_landmarks;
          }
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::addScaleImpl() {

      utils::ScopedTimer<scalar_type, utils::Seconds> timer_tot(_statistics._total_time);
      utils::secureLog(_logger, "Add a new scale ...");

      _hierarchy.push_back(Scale());
      Scale& scale = _hierarchy[_hierarchy.size() - 1];
      Scale& previous_scale = _hierarchy[_hierarchy.size() - 2];

      const unsigned_int_type previous_scale_dp = previous_scale._landmark_to_original_data_idx.size();

      // Landmark selection
      unsigned_int_type selected_landmarks = 0;
      if (_params._monte_carlo_sampling) {
        selectLandmarksWithStationaryDistribution(previous_scale, scale, selected_landmarks);
      }
      else {
        selectLandmarks(previous_scale, scale, selected_landmarks);
      }

      utils::secureLogValue(_logger, "\t#landmarks", selected_landmarks);

      {//Area of influence
        //__block std::default_random_engine generator(seed());
        //__block std::uniform_real_distribution<double> distribution_real(0.0, 1.0);
        std::default_random_engine generator(seed());
        std::uniform_real_distribution<double> distribution_real(0.0, 1.0);
        const unsigned_int_type max_jumps = 100;//1000.*selected_landmarks/previous_scale_dp;
        const unsigned_int_type walks_per_dp = _params._num_walks_per_landmark;
        utils::secureLog(_logger, "\tComputing area of influence...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aoi_time);
          //__block unsigned_int_type num_elem_in_Is(0);
          unsigned_int_type num_elem_in_Is(0);
          //#ifdef __USE_GCD__
          //          std::cout << "GCD dispatch, hierarchical_sne_inl 473.\n";
          //          dispatch_queue_t criticalQueue = dispatch_queue_create("critical", NULL);
          //          dispatch_apply(previous_scale_dp, dispatch_get_global_queue(0, 0), ^(size_t d) {
          //#else
#pragma omp parallel for
          for (int_type d = 0; d < previous_scale_dp; ++d) {
            //#endif //__USE_GCD__
            std::unordered_map<unsigned_int_type, unsigned_int_type> landmarks_reached;
            for (unsigned_int_type i = 0; i < walks_per_dp; ++i) {
              auto res = randomWalk(d, scale._previous_scale_to_landmark_idx, max_jumps, previous_scale._transition_matrix, distribution_real, generator);
              if (res != -1) {
                ++landmarks_reached[scale._previous_scale_to_landmark_idx[res]];
              }
              else {
                //--i;
              }
            }


            //#ifdef __USE_GCD__
            //            dispatch_sync(criticalQueue, ^
            //#else
#pragma omp critical
//#endif
            {
              num_elem_in_Is += landmarks_reached.size();

              for (const auto& l : landmarks_reached) {
                for (const auto& other_l : landmarks_reached) {
                  //to avoid that the sparsity of the matrix it is much different from the effective sparsity
                  if (l.second <= _params._transition_matrix_prune_thresh || other_l.second <= _params._transition_matrix_prune_thresh)
                    continue;
                  if (l.first != other_l.first) {
                    scale._transition_matrix[l.first][other_l.first] += l.second * other_l.second * previous_scale._landmark_weight[d];
                  }
                }
              }

              for (const auto& l : landmarks_reached) {
                const scalar_type prob = scalar_type(l.second) / walks_per_dp;
                scale._area_of_influence[d][l.first] = prob;
                scale._landmark_weight[l.first] += prob * previous_scale._landmark_weight[d];
              }
            }
            //#ifdef __USE_GCD__
            //            );
            //#endif
          }
          //#ifdef __USE_GCD__
          //          );
          //#endif
          _statistics._aoi_num_walks = previous_scale_dp * walks_per_dp;
          _statistics._aoi_sparsity = 1 - scalar_type(num_elem_in_Is) / (previous_scale_dp*selected_landmarks);
        }

        {
          utils::secureLog(_logger, "\tComputing finite markov chain...");
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._fmc_time);
          unsigned_int_type num_elem_in_Ts(0);
          unsigned_int_type num_effective_elem_in_Ts(0);
          for (size_t l = 0; l < scale._transition_matrix.size(); ++l) {
            num_elem_in_Ts += scale._transition_matrix[l].size();
            scalar_type sum(0);
            for (auto& e : scale._transition_matrix[l]) {
              sum += e.second;
            }
            for (auto& e : scale._transition_matrix[l]) {
              e.second /= sum;
              if (e.second > 0.01) {
                ++num_effective_elem_in_Ts;
              }
            }
          }
          _statistics._fmc_sparsity = 1 - scalar_type(num_elem_in_Ts) / (selected_landmarks*selected_landmarks);
          _statistics._fmc_effective_sparsity = 1 - scalar_type(num_effective_elem_in_Ts) / (selected_landmarks*selected_landmarks);
        }
      }

      utils::secureLogValue(_logger, "Min memory requirements (MB)", scale.mimMemoryOccupation());

      return true;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::addScaleOutOfCoreImpl() {
      typedef hdi::data::MapHelpers<map_key_type, map_value_type, map_type> map_helpers_type;

      utils::ScopedTimer<scalar_type, utils::Seconds> timer_tot(_statistics._total_time);
      utils::secureLog(_logger, "Add a new scale with out-of-core implementation ...");

      _hierarchy.push_back(Scale());
      Scale& scale = _hierarchy[_hierarchy.size() - 1];
      Scale& previous_scale = _hierarchy[_hierarchy.size() - 2];

      const unsigned_int_type previous_scale_dp = previous_scale._landmark_to_original_data_idx.size();

      // Landmark selection
      unsigned_int_type selected_landmarks = 0;
      if (_params._monte_carlo_sampling) {
        selectLandmarksWithStationaryDistribution(previous_scale, scale, selected_landmarks);
      }
      else {
        selectLandmarks(previous_scale, scale, selected_landmarks);
      }

      utils::secureLogValue(_logger, "\t#landmarks", selected_landmarks);

      {//Area of influence
        std::default_random_engine generator(seed());
        std::uniform_real_distribution<double> distribution_real(0.0, 1.0);
        const unsigned_int_type max_jumps = 200;//1000.*selected_landmarks/previous_scale_dp;
        const unsigned_int_type walks_per_dp = _params._num_walks_per_landmark;
        utils::secureLog(_logger, "\tComputing area of influence...");
        {
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aoi_time);
          unsigned_int_type num_elem_in_Is(0);

          {
            utils::LogProgress progress(_verbose ? _logger : nullptr);
            progress.setNumSteps(previous_scale_dp);
            progress.setNumTicks(previous_scale_dp / 50000);
            progress.setName("Area of influence");
            progress.start();
            //#ifdef __USE_GCD__
            //          std::cout << "GCD dispatch, hierarchical_sne_inl 587.\n";
            //          dispatch_queue_t criticalQueue = dispatch_queue_create("critical", NULL);
            //          dispatch_apply(previous_scale_dp, dispatch_get_global_queue(0, 0), ^(size_t d) {
            //#else
#pragma omp parallel for
            for (int_type d = 0; d < previous_scale_dp; ++d) {
              //#endif //__USE_GCD__
                          //map because it must be ordered for the initialization of the maps
              std::map<map_key_type, scalar_type> landmarks_reached;
              for (unsigned_int_type i = 0; i < walks_per_dp; ++i) {
                auto res = randomWalk(d, scale._previous_scale_to_landmark_idx, max_jumps, previous_scale._transition_matrix, distribution_real, generator);
                if (res != -1) {
                  ++landmarks_reached[scale._previous_scale_to_landmark_idx[res]];
                }
                else {
                  //--i;
                }
              }

              //normalization
              for (auto& l : landmarks_reached) {
                l.second = scalar_type(l.second) / walks_per_dp;
              }
              //saving aoi
              map_helpers_type::initialize(scale._area_of_influence[d], landmarks_reached.begin(), landmarks_reached.end());
              map_helpers_type::shrinkToFit(scale._area_of_influence[d]);
              progress.step();
            }
            //#ifdef __USE_GCD__
            //          );
            //#endif
            progress.finish();
          }
          utils::secureLog(_logger, "\tCaching weights...");
          //caching of the weights
          for (unsigned_int_type d = 0; d < previous_scale_dp; ++d) {
            num_elem_in_Is += scale._area_of_influence[d].size();
            for (auto& e : scale._area_of_influence[d]) {
              scale._landmark_weight[e.first] += e.second;
            }
          }


          utils::secureLog(_logger, "\tInverting the AoI matrix...");
          //Inverse AoI -> critical for the computation time
          sparse_scalar_matrix_type inverse_aoi = {};
          map_helpers_type::invert(scale._area_of_influence, inverse_aoi);

          utils::secureLog(_logger, "\tComputing similarities...");
          //Similarities -> compute the overlap of the area of influence

          {
            utils::LogProgress progress(_verbose ? _logger : nullptr);
            progress.setNumSteps(scale._transition_matrix.size());
            progress.setNumTicks(scale._transition_matrix.size() / 5000);
            progress.setName("Similarities");
            progress.start();
            //  #ifdef __USE_GCD__
            //            std::cout << "GCD dispatch, hierarchical_sne_inl 602.\n";
            //            dispatch_apply(scale._transition_matrix.size(), dispatch_get_global_queue(0, 0), ^(size_t l) {
            //  #else
#pragma omp parallel for
            for (int_type l = 0; l < scale._transition_matrix.size(); ++l) {
              //  #endif //__USE_GCD__
                            //ordered for efficient initialization
              std::map<map_key_type, map_value_type> temp_trans_mat; // use map here
              for (const auto& d : inverse_aoi[l]) {
                for (const auto& aoi : scale._area_of_influence[d.first]) {
                  double single_landmark_thresh = (1. / 100.)*_params._transition_matrix_prune_thresh;
                  if (l != aoi.first) {
                    if (d.second <= single_landmark_thresh || aoi.second <= single_landmark_thresh)
                      continue;
                    temp_trans_mat[aoi.first] += d.second * aoi.second * previous_scale._landmark_weight[d.first];
                  }
                }
              }
              //normalization
              double sum = 0;
              for (const auto& v : temp_trans_mat) { sum += v.second; }
              for (auto& v : temp_trans_mat) { v.second /= sum; }

              //removed the threshold depending on the scale -> it makes sense to remove only uneffective neighbors based at every scale -> memory is still under control
              map_helpers_type::initialize(scale._transition_matrix[l],temp_trans_mat.begin(),temp_trans_mat.end(), static_cast<map_value_type>(0.001));
              map_helpers_type::shrinkToFit(scale._transition_matrix[l]);
              progress.step();
            }
            //  #ifdef __USE_GCD__
            //            );
            //  #endif
            progress.finish();
          }
          _statistics._aoi_num_walks = previous_scale_dp * walks_per_dp;
          _statistics._aoi_sparsity = 1 - scalar_type(num_elem_in_Is) / (previous_scale_dp*selected_landmarks);
        }

        {
          utils::secureLog(_logger, "\tComputing finite markov chain...");
          utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._fmc_time);
          unsigned_int_type num_elem_in_Ts(0);
          unsigned_int_type num_effective_elem_in_Ts(0);
          for (size_t l = 0; l < scale._transition_matrix.size(); ++l) {
            num_elem_in_Ts += scale._transition_matrix[l].size();
            scalar_type sum(0);
            for (const auto& e : scale._transition_matrix[l]) {
              sum += e.second;
            }
            for (auto& e : scale._transition_matrix[l]) {
              e.second /= sum;
              if (e.second > 0.001) {
                ++num_effective_elem_in_Ts;
              }
            }
          }
          _statistics._fmc_sparsity = 1 - scalar_type(num_elem_in_Ts) / (selected_landmarks*selected_landmarks);
          _statistics._fmc_effective_sparsity = 1 - scalar_type(num_effective_elem_in_Ts) / (selected_landmarks*selected_landmarks);
        }
      }

      return true;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    typename HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::unsigned_int_type HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::seed()const {
      return(_params._seed > 0) ? static_cast<unsigned_int_type>(_params._seed) : std::chrono::system_clock::now().time_since_epoch().count();
    }


    ///////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getInfluencedLandmarksInPreviousScale(unsigned_int_type scale_id, std::vector<unsigned_int_type>& idxes, std::map<unsigned_int_type, scalar_type>& neighbors)const {
      neighbors.clear();
      std::unordered_set<unsigned_int_type> set_idxes;
      set_idxes.insert(idxes.begin(), idxes.end());
      auto not_found = set_idxes.end();

      // For each data point (landmark L_{i}) in the previous (s-1) scale...
      for (size_t d = 0; d < _hierarchy[scale_id]._area_of_influence.size(); ++d) {
        double probability = 0;
        // ... check if any landmark index at this scale (s) affects it
        for (auto& v : _hierarchy[scale_id]._area_of_influence[d]) {
          // ... if so sum the probability
          // Performs: https://doi.org/10.1111/cgf.12878 4.2 eqn 8 to calculate total influence F_i 
          // on landmark L_{i}^{s-1} for selection group O (idxes) (follow link to visualize equation)
          // https://latex.codecogs.com/png.image?F_i=\sum_{\mathcal{L}_{j}^{s}\in\mathcal{O}}I^S(i,j)
          if (set_idxes.find(v.first) != not_found) {
            probability += v.second;
          }
        }
        // Record that s-1 landmark index is a neigbor of this idxes set
        // and the total probability F_i for landmark L_{i}^{s-1}
        if (probability > 0) {
          neighbors[d] = probability;
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getInfluencingLandmarksInNextScale(unsigned_int_type scale_id, std::vector<unsigned_int_type>& idxes, std::map<unsigned_int_type, scalar_type>& neighbors)const {

      neighbors.clear();

      unsigned_int_type next_scale_id = scale_id + 1;
      if (next_scale_id + 1 > _hierarchy.size()) return;

      std::map<unsigned_int_type, scalar_type> completeSet;

      for (size_t i = 0; i < idxes.size(); i++)
      {
        for (auto& v : _hierarchy[next_scale_id]._area_of_influence[idxes[i]]) {
          neighbors[v.first] += v.second;
        }
      }

      for (size_t i = 0; i < _hierarchy[next_scale_id]._area_of_influence.size(); i++)
      {
        for (auto& v : _hierarchy[next_scale_id]._area_of_influence[i]) {
          completeSet[v.first] += v.second;
        }
      }

      for (auto& v : neighbors)
      {
        neighbors[v.first] /= completeSet[v.first];
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getInterpolationWeights(sparse_scalar_matrix_type& influence, int scale)const {
      influence.clear();
      influence.resize(_num_dps);

      scale = (scale < 0) ? (_hierarchy.size() - 1) : scale;
      checkAndThrowLogic(scale < _hierarchy.size(), "getInterpolationWeights: Invalid scale");

      //#ifdef __USE_GCD__
      //      std::cout << "GCD dispatch, hierarchical_sne_inl 724.\n";
      //      dispatch_apply(_num_dps, dispatch_get_global_queue(0, 0), ^(size_t i) {
      //#else
#pragma omp parallel for
      for (int_type i = 0; i < _num_dps; ++i) {
        //#endif //__USE_GCD__
        influence[i] = _hierarchy[1]._area_of_influence[i];
        for (int s = 2; s <= scale; ++s) {
          typename sparse_scalar_matrix_type::value_type temp_link = {};
          for (auto l : influence[i]) {
            for (auto new_l : _hierarchy[s]._area_of_influence[l.first]) {
              temp_link[new_l.first] += l.second * new_l.second;
            }
          }
          influence[i] = temp_link;
        }
      }
      //#ifdef __USE_GCD__
      //      );
      //#endif
    }
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getInterpolationWeights(const std::vector<unsigned int>& data_points, sparse_scalar_matrix_type& influence, int scale)const {
      const size_t n = data_points.size();
      influence.clear();
      influence.resize(n);

      scale = (scale < 0) ? (_hierarchy.size() - 1) : scale;
      checkAndThrowLogic(scale < _hierarchy.size(), "getInterpolationWeights: Invalid scale");

      //#ifdef __USE_GCD__
      //      std::cout << "GCD dispatch, hierarchical_sne_inl 755.\n";
      //      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^(size_t i) {
      //#else
#pragma omp parallel for
      for (int_type i = 0; i < n; ++i) {
        //#endif //__USE_GCD__
        influence[i] = _hierarchy[1]._area_of_influence[data_points[i]];
        for (int s = 2; s <= scale; ++s) {
          typename sparse_scalar_matrix_type::value_type temp_link = {};
          for (const auto& l : influence[i]) {
            for (const auto& new_l : _hierarchy[s]._area_of_influence[l.first]) {
              temp_link[new_l.first] += l.second * new_l.second;
            }
          }
          influence[i] = temp_link;
        }
      }
      //#ifdef __USE_GCD__
      //      );
      //#endif
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getInfluenceOnDataPoint(unsigned_int_type dp, std::vector<std::unordered_map<unsigned_int_type, scalar_type>>& influence, scalar_type thresh, bool normalized)const {
      assert(dp < _hierarchy[0].size());
      influence.resize(_hierarchy.size());
      influence[0][dp] = 1; //Hey it's me!
      if (influence.size() == 1) {
        return;
      }

      for (const auto& v : _hierarchy[1]._area_of_influence[dp]) {
        influence[1][v.first] = v.second;
      }
      if (normalized)
      {
        double sum = 0;
        for (const auto& v : influence[1]) { sum += v.second; }
        for (auto& v : influence[1]) { v.second /= sum; }
      }
      for (size_t s = 2; s < _hierarchy.size(); ++s) {
        for (const auto& l : influence[s - 1]) {
          if (l.second >= thresh) {
            for (auto new_l : _hierarchy[s]._area_of_influence[l.first]) {
              influence[s][new_l.first] += l.second * new_l.second;
            }
          }
        }
        if (normalized)
        {
          double sum = 0;
          for (const auto& v : influence[s]) { sum += v.second; }
          for (auto& v : influence[s]) { v.second /= sum; }
        }
      }
    }

    /**
     * @brief Given a data point index return the landmark influence vector for all scales.
     *  Optionally threshold and optionally normalize the influences.
     * @details
	   *  The influence vector (parameter influence) contains a map for each scale containing the influence
	   *  value for each landmark (data point) at that scale to the given data point (dp).
     *
     *  At scale 0 the influence vector has a single entry with index being data point index and value 1
     *  At scale 1 (the first landmark scale) the map contains an entry for all landmarks with influence values
	   *  At scales above scale 1 - i.e. s:
	   *		For every landmark in the s-1 influence map if the influence exceeds the threshold
	   *		multiple that infuence by the aoi values at scale s and include it
     *
	   * Note _area_of_influence at hierarchy scale s has size(s-1) entries containing a 
     * list of tuples of landmark index and its influence the corresponding s-1 point.
     *
     * @tparam scalar_type used for influence value
     * @tparam sparse_scalar_matrix_type used for landmark(s-1) -> landmark(s) + influence
     * @param dp Data point index
     * @param max_landmark_per_scale Landmark influence vector for all scales (returned)
     * @param scale_has_landmark If a landmark was fond exceeding the threshold the scale entry is true
     * @param thresh Optional threshold - default is 0.
     * @param normalized Optional normalization - default is false
     */
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getTopLandmarksInfluencingDataPoint(
        unsigned_int_type dp,
        std::vector<unsigned_int_type>& top_landmark_per_scale,
        std::vector<bool>& scale_has_landmark,
        scalar_type thresh,
        bool normalized) const {
      // a valid data point index is in the size of data points
      assert(dp < _hierarchy[0].size());

      top_landmark_per_scale.resize(_hierarchy.size());
      scale_has_landmark.resize(_hierarchy.size());

      std::vector<std::unordered_map<unsigned_int_type, scalar_type>> influence(_hierarchy.size());
      influence[0][dp] = 1; //Only this point influences itself at the data level.
      if (influence.size() == 1) {
        return;
      }

      scalar_type maxInfluence = 0;
      for (const auto& v : _hierarchy[1]._area_of_influence[dp]) {
        influence[1][v.first] = v.second;
        if (maxInfluence < v.second) {
          maxInfluence = v.second;
          scale_has_landmark[1] = true;
          top_landmark_per_scale[1] = v.first;
        }
      }

      if (normalized) {
        double sum = 0;
        for (const auto& v : influence[1]) { sum += v.second; }
        for (auto& v : influence[1]) { v.second /= sum; }
      }

      for (size_t s = 2; s < _hierarchy.size(); ++s) {
        for (const auto& l : influence[s - 1]) {
          maxInfluence = 0;
          if (l.second >= thresh) {
            for (auto new_l : _hierarchy[s]._area_of_influence[l.first]) {
              influence[s][new_l.first] += l.second * new_l.second;
              if (maxInfluence < influence[s][new_l.first]) {
                maxInfluence = influence[s][new_l.first];
                scale_has_landmark[s] = true;
                top_landmark_per_scale[s] = new_l.first;
              }
            }
          }
        }
        if (normalized) {
          double sum = 0;
          for (auto& v : influence[s]) { sum += v.second; }
          for (auto& v : influence[s]) { v.second /= sum; }
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::initializeScaleLandmarkToDataPointHierarchy()
    {
      // Create an empty data structure with a empty 
      // data point vector space for all
      // landmarks in all scales
      const size_t num_scales     = _hierarchy.size();
      const size_t num_datapoints = _hierarchy[0].size();
      _landmarks_to_datapoints.resize(num_scales);

      #pragma omp parallel for
      for (int_type scale = 1; scale < num_scales; scale++) {
        const size_t num_landmarks = _hierarchy[scale].size();
        _landmarks_to_datapoints[scale].resize(num_landmarks);

        for (size_t landmark = 0; landmark < num_landmarks; landmark++) {
          _landmarks_to_datapoints[scale][landmark].clear();
          // The reserve assumes that landmark will never have more
          // than twice the average number of data points per landmark.
          // Where is this guaranteed?
          _landmarks_to_datapoints[scale][landmark].reserve(2 * num_datapoints/num_landmarks);
        }
      }

      #pragma omp parallel for
      for(int_type dp=0; dp < num_datapoints; dp++) {
        scalar_type inf_thresh = 0.01;
        std::vector<unsigned_int_type> top_landmark_per_scale;
        std::vector<bool> scale_has_landmark;
        getTopLandmarksInfluencingDataPoint(dp, top_landmark_per_scale, scale_has_landmark, inf_thresh, false);

        int redo = 1;
        int tries = 0;
        while (redo) {
          redo = 0;
          if (tries++ < 3) {
            for (int scale = 1; scale < num_scales; scale++) {
              if (!scale_has_landmark[scale]) {
                redo = scale;
              }
            }
          }
          if (redo > 0) {
            //if(thresh < 0.0005) std::cout << "Couldn't find landmark for point " << i << " at scale " << redo << " num possible landmarks " << influence[redo].size() << "\nSetting new threshold to " << thresh * 0.1 << std::endl;
            inf_thresh *= 0.1;
            getTopLandmarksInfluencingDataPoint(dp, top_landmark_per_scale, scale_has_landmark, inf_thresh, false);
          }
        }
        for (size_t scale = 1; scale < num_scales; scale++) {
          #pragma omp critical
          {
            _landmarks_to_datapoints[scale][top_landmark_per_scale[scale]].push_back(dp);
          }
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getStochasticLocationAtHigherScale(unsigned_int_type orig_scale, unsigned_int_type dest_scale, const std::vector<unsigned_int_type>& subset_orig_scale, sparse_scalar_matrix_type& closeness)const {
      checkAndThrowLogic(dest_scale > orig_scale, "getStochasticLocationAtHigherScale (0)");
      checkAndThrowLogic(orig_scale < _hierarchy.size() - 1, "getStochasticLocationAtHigherScale (2)");
      checkAndThrowLogic(dest_scale < _hierarchy.size(), "getStochasticLocationAtHigherScale (3)");

      closeness.clear();
      closeness.resize(subset_orig_scale.size());

      //#ifdef __USE_GCD__
      //      std::cout << "GCD dispatch, hierarchical_sne_inl 814.\n";
      //      dispatch_apply(subset_orig_scale.size(), dispatch_get_global_queue(0, 0), ^(size_t i) {
      //#else
#pragma omp parallel for
      for (int_type i = 0; i < subset_orig_scale.size(); ++i) {
        //#endif //__USE_GCD__
        assert(subset_orig_scale[i] < _hierarchy[orig_scale + 1]._area_of_influence.size());
        closeness[i] = _hierarchy[orig_scale + 1]._area_of_influence[subset_orig_scale[i]];

        for (unsigned_int_type s = orig_scale + 2; s <= dest_scale; ++s) {
          typename sparse_scalar_matrix_type::value_type temp_link = {};
          for (const auto& l : closeness[i]) {
            for (const auto& new_l : _hierarchy[s]._area_of_influence[l.first]) {
              temp_link[new_l.first] += l.second * new_l.second;
            }
          }
          closeness[i] = temp_link;
        }
      }
      //#ifdef __USE_GCD__
      //      );
      //#endif
    }


    //! This function computes the cumulative area of influence (aoi) of a "selection"of  landmark points at scale "scale_id" for each point at the data level. 
    //! This process is described in Section 3.3 of https://doi.org/10.1111/cgf.12878
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getAreaOfInfluence(unsigned_int_type scale_id, const std::vector<unsigned_int_type>& selection, std::vector<scalar_type>& aoi)const {
      typedef hdi::data::MapHelpers<map_key_type, map_value_type, map_type> map_helpers_type;
      checkAndThrowLogic(scale_id < _hierarchy.size(), "getAreaOfInfluence (3)");

	  // initialize the area of influence vector
      aoi.assign(scale(0).size(), 0);

	  // at scale 0 every point has a maximum area of influence (1) on itself.
      if (scale_id == 0) {
		#pragma omp parallel for 
        for (int_type i = 0; i < selection.size(); ++i) {
          aoi[selection[i]] = 1;
        }
      }
      else {
		  const std::unordered_set<unsigned_int_type> selected_scale_id_landmarks(selection.cbegin(), selection.cend()); // unordered set since we need all items to be unique 
		  // Compute for every point at the data level the area of influence (aoi) 
		  // of the "selection" landmark points at scale scale_id through a chain of sparse matrix multiplications
		  #pragma omp parallel for schedule(dynamic,1)
		  for (int_type i = 0; i < scale(0).size(); ++i) {
			  const auto& scale_1_aois = scale(1)._area_of_influence[i];
			  // Declare a holder for the super scale landmarks aois, using std::vector for quick look-up 
			  std::vector<std::pair<map_key_type, map_value_type>> super_aois(scale_1_aois.begin(), scale_1_aois.end());
			  // Walk the scale hierarchy from super-scale to sub-scale
			  // For each scale compute the cumulative influence 
			  // of the landmark points from the lower scale on data point "i" at the current scale.
			  for (unsigned_int_type s = 2; s <= scale_id; ++s) {
				  // Use unordered_map for quick insertion
				  // Ordering is not needed but we do want to avoid multiple entries per data/landmark point.
				  std::unordered_map<map_key_type, map_value_type> current_aoi_cumulative;
				  // Factor in the influence of all the super scale landmark aois
				  for (auto super_aoi : super_aois) {
					  for (auto current_aoi : scale(s)._area_of_influence[super_aoi.first]) {
						  // compute cumulative aoi sum of products 
						  current_aoi_cumulative[current_aoi.first] += super_aoi.second * current_aoi.second;
					  }
				  }
				  // Copy the influence of the current_aoi_cumulative to the super scale aoi holder for the next iteration
				  super_aois.resize(current_aoi_cumulative.size());
				  std::copy(current_aoi_cumulative.cbegin(), current_aoi_cumulative.cend(), super_aois.begin());
			  }
			  // Now the current scale for super_aois is scale_id.
			  // So the indices in "selection" match the indices in super_aois
			  // Check if that landmark is in the selection,
			  // if so add the area of influence to the cumulative area of influence of the selection on point i
			  for (auto scale_id_aoi : super_aois) {
				  if (selected_scale_id_landmarks.find(scale_id_aoi.first) != selected_scale_id_landmarks.end()) {
					  aoi[i] += scale_id_aoi.second;
				  }
			  }
		  }
	  }
	}

    //! This function computes the cumulative area of influence (aoi) of a "selection" 
    //! of landmark points at scale "scale_id" for each point at the data level. 
    //! This process is described in Section 4.2 of https://doi.org/10.1111/cgf.12878
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getAreaOfInfluenceTopDown(unsigned_int_type scale_id, const std::vector<unsigned_int_type>& selection, std::vector<scalar_type>& aoi, double threshold) const {
      checkAndThrowLogic(scale_id < _hierarchy.size(), "getAreaOfInfluenceTopDown (3)");

      double gamma = 0.3;
      if ((threshold >= 0) && (threshold <= 1.0)) {
        gamma = threshold;
      }
      aoi.clear();
      aoi.resize(scale(0).size(), 0);
      std::unordered_set<unsigned_int_type> set_selected_idxes;
      set_selected_idxes.insert(selection.begin(), selection.end());

      if (scale_id == 0) {
        for (size_t i = 0; i < selection.size(); ++i) {
          aoi[selection[i]] = 1;
        }
      }
      else {

        std::vector<unsigned_int_type> scale_selection = selection;
        for (unsigned_int_type s = scale_id; s > 0; --s) {
          std::map<unsigned_int_type, scalar_type> neighbors;
          getInfluencedLandmarksInPreviousScale(s, scale_selection, neighbors);
          scale_selection.clear();
          for (const auto& neigh : neighbors) {
            if (neigh.second > gamma) {
              scale_selection.push_back(neigh.first);
            }
          }
        }
        for (size_t i = 0; i < scale_selection.size(); ++i) {
          aoi[scale_selection[i]] = 1;
        }
      }
    }

    //! This function computes the cumulative area of influence (aoi) of a "selection"
    //! of landmark points at scale "scale_id" for each point at the data level.
    //!
    //! The mapping of scale landmarks to datapoints is stored in _landmarks_to_datapoints
    //! which is set in the initializeScaleLandmarkToDataPointHierarchy function
    //! _landmarks_to_datdapoints is created by walking the scale hierarchy
    //! bottom up and choosing the landmark that best represents the points at the scale below.
    //!
    //! This function will check if _landmarks_to_datapoints has been initialized and if not
    //! will call initializeScaleLandmarkToDataPointHierarchy
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::getAreaOfInfluenceBottomUp(
      unsigned_int_type scale_id,
      const std::vector<unsigned_int_type>& selection,
      std::vector<scalar_type>& aoi) {
      auto num_scales = _hierarchy.size();
      if (_landmarks_to_datapoints.size() == 0) {
        initializeScaleLandmarkToDataPointHierarchy();
      }
      checkAndThrowLogic(_landmarks_to_datapoints.size(), "Landmark -> datapoint mapping not initialized!");
      aoi.clear();
      aoi.resize(scale(0).size(), 0);
      const auto& scale_landmark_to_dp = _landmarks_to_datapoints[scale_id];
      for (const auto landmark: selection) {
        for (size_t i = 0; i < scale_landmark_to_dp[landmark].size(); ++i) {
          aoi[scale_landmark_to_dp[landmark][i]] = 1;
        }
      }
    }

    ///////////////////////////////////////////////////////////////////
    /// RANDOM WALKS
    ///////////////////////////////////////////////////////////////////

        //Compute a random walk using a transition matrix
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    typename HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::unsigned_int_type HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::randomWalk(unsigned_int_type starting_point, unsigned_int_type max_length, const sparse_scalar_matrix_type& transition_matrix, std::uniform_real_distribution<double>& distribution, std::default_random_engine& generator) {
      unsigned_int_type dp_idx = starting_point;
      unsigned_int_type walk_length = 0;
      do {
        const double rnd_num = distribution(generator);
        unsigned_int_type idx_knn = dp_idx;
        double incremental_prob = 0;
        for (const auto& elem : transition_matrix[dp_idx]) {
          incremental_prob += elem.second;
          if (rnd_num < incremental_prob) {
            idx_knn = elem.first;
            break;
          }
        }
        //assert(idx_knn != dp_idx);
        if (idx_knn == dp_idx) {
          return std::numeric_limits<unsigned_int_type>::max();
          //          std::cout << "DISCONNECTED!" << std::endl;
        }
        dp_idx = idx_knn;
        ++walk_length;
      } while (walk_length <= max_length);
      return dp_idx;
    }

    //!Compute a random walk using a transition matrix
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    int HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::randomWalk(unsigned_int_type starting_point, const std::vector<int_type>& stopping_points, unsigned_int_type max_length, const sparse_scalar_matrix_type& transition_matrix, std::uniform_real_distribution<double>& distribution, std::default_random_engine& generator) {
      unsigned_int_type dp_idx = starting_point;
      unsigned_int_type walk_length = 0;
      do {
        const double rnd_num = distribution(generator);
        unsigned_int_type idx_knn = dp_idx;
        double incremental_prob = 0;
        for (const auto& elem : transition_matrix[dp_idx]) {
          incremental_prob += elem.second;
          if (rnd_num < incremental_prob) {
            idx_knn = elem.first;
            break;
          }
        }
        //assert(idx_knn != dp_idx);
        if (idx_knn == dp_idx) {
          return -1;
          std::cout << "42!" << std::endl;
        }
        dp_idx = idx_knn;
        ++walk_length;
      } while (stopping_points[dp_idx] == -1 && walk_length <= max_length);
      if (walk_length > max_length) {
        return -1;
      }
      return static_cast<int>(dp_idx);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    typename HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::int_type HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::getFreeClusterId(unsigned_int_type scale_id) {
      int_type max = std::numeric_limits<int_type>::max();
      for (int_type i = 0; i < max; ++i) {
        for (size_t j = 0; j < _cluster_tree[scale_id].size(); ++j) {
          if (i != _cluster_tree[scale_id][j].id()) {
            return i;
          }
        }
      }
      return 0;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::addCluster(unsigned_int_type scale_id, const cluster_type& cluster) {
      checkAndThrowLogic(scale_id < _cluster_tree.size(), "ClusterHierarchy::addCluster: invalid scale");
      for (size_t j = 0; j < _cluster_tree[scale_id].size(); ++j) {
        checkAndThrowLogic(cluster.id() != _cluster_tree[scale_id][j].id(), "ClusterHierarchy::addCluster: duplicated id");
      }
      if (scale_id == _cluster_tree.size() - 1) {
        checkAndThrowLogic(cluster.parent_id() == Cluster::NULL_LINK, "ClusterHierarchy::addCluster: root clusters must have parent_id = Cluster::NULL_LINK");
      }
      else {
        checkAndThrowLogic(cluster.parent_id() != Cluster::NULL_LINK, "ClusterHierarchy::addCluster: non-root clusters must have parent_id != Cluster::NULL_LINK");
      }

      _cluster_tree[scale_id].push_back(cluster);
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::removeCluster(unsigned_int_type scale_id, int_type cluster_id) {
      checkAndThrowLogic(scale_id < _cluster_tree.size(), "ClusterHierarchy::removeCluster: invalid scale");
      for (size_t i = 0; i < _cluster_tree[scale_id].size(); ++i) {
        if (_cluster_tree[scale_id][i].id() == cluster_id) {
          _cluster_tree[scale_id].erase(_cluster_tree[scale_id].begin() + i);
          break;
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::hasClusterId(unsigned_int_type scale_id, int_type cluster_id)const {
      checkAndThrowLogic(scale_id < _cluster_tree.size(), "ClusterHierarchy::hasClusterId: invalid scale");
      for (size_t j = 0; j < _cluster_tree[scale_id].size(); ++j) {
        if (cluster_id == _cluster_tree[scale_id][j].id()) { return true; }
      }
      return false;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    const typename HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::cluster_type& HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::cluster(unsigned_int_type scale_id, int_type cluster_id)const {
      checkAndThrowLogic(hasClusterId(scale_id, cluster_id), "ClusterHierarchy::cluster: invalid cluster");
      for (size_t j = 0; j < _cluster_tree[scale_id].size(); ++j) {
        if (cluster_id == _cluster_tree[scale_id][j].id()) {
          return _cluster_tree[scale_id][j];
        }
      }
      throw std::logic_error("Invalid cluster");
      //return cluster_type(); //INVALID
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::checkCluterConsistency(const HierarchicalSNE& hsne, unsigned_int_type scale_id, int_type cluster_id) {
      checkAndThrowLogic(hasClusterId(scale_id, cluster_id), "ClusterHierarchy::checkCluterConsistency: invalid cluster");
      if (scale_id == _cluster_tree.size() - 1) {
        std::stringstream ss;
        ss << "Validating cluster " << cluster_id << " at scale " << scale_id << ":\tis a root node => valid";
        utils::secureLog(_logger, ss.str());
        return true;
      }
      int_type cluster_id_in_vector = -1;
      for (size_t j = 0; j < _cluster_tree[scale_id].size(); ++j) {
        if (cluster_id == _cluster_tree[scale_id][j].id()) {
          cluster_id_in_vector = j;
        }
      }

      std::vector<scalar_type> influence(_cluster_tree[scale_id + 1].size(), 0);
      scalar_type unclustered_influence(0);

      auto& scale = hsne.scale(scale_id + 1);

      for (const auto& e : _cluster_tree[scale_id][cluster_id_in_vector].landmarks()) {
        for (const auto& aoi : scale._area_of_influence[e]) {
          bool found = false;
          for (size_t i = 0; i < influence.size(); ++i) {
            auto it = _cluster_tree[scale_id + 1][i].landmarks().find(aoi.first);
            if (it != _cluster_tree[scale_id + 1][i].landmarks().end()) {
              influence[i] += aoi.second;
              found = true;
            }
          }
          if (!found) {
            unclustered_influence += aoi.second;
          }
        }
      }

      std::stringstream ss;
      ss << "Validating cluster " << cluster_id << " at scale " << scale_id << " with parent " << _cluster_tree[scale_id][cluster_id_in_vector].parent_id() << " (" << _cluster_tree[scale_id][cluster_id_in_vector].notes() << ")" << std::endl;
      ss << "\tUnclusterd:\t" << unclustered_influence << std::endl;
      scalar_type max(unclustered_influence);
      int_type res_id(-1);
      for (size_t i = 0; i < influence.size(); ++i) {
        ss << "\tCluster-" << _cluster_tree[scale_id + 1][i].id() << " (" << _cluster_tree[scale_id + 1][i].notes() << ") :\t" << influence[i] << std::endl;
        if (influence[i] > max) {
          max = influence[i];
          res_id = _cluster_tree[scale_id + 1][i].id();
        }
      }
      utils::secureLog(_logger, ss.str());

      if (res_id == _cluster_tree[scale_id][cluster_id_in_vector].parent_id()) {
        utils::secureLog(_logger, "Valid");
        return true;
      }
      utils::secureLog(_logger, "INVALID!");
      return false;
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    bool HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::checkTreeConsistency(const HierarchicalSNE& hsne) {
      bool res = true;
      for (int s = _cluster_tree.size() - 1; s >= 0; --s) {
        for (int c = 0; c < _cluster_tree[s].size(); ++c) {
          res &= checkCluterConsistency(hsne, s, _cluster_tree[s][c].id());
        }
      }
      return res;
    }


    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::computePointToClusterAssociation(const HierarchicalSNE& hsne, unsigned_int_type pnt_id, std::tuple<unsigned_int_type, int_type, scalar_type>& res) {
      std::vector<std::unordered_map<unsigned_int_type, scalar_type>> influence;
      hsne.getInfluenceOnDataPoint(pnt_id, influence);

      res = std::tuple<unsigned_int_type, int_type, scalar_type>(_cluster_tree.size() - 1ll, -1ll, 1.f);
      std::vector<unsigned_int_type> clusters_to_analyze(_cluster_tree[_cluster_tree.size() - 1].size());
      std::iota(clusters_to_analyze.begin(), clusters_to_analyze.end(), 0);

      //just for test
      for (size_t s = _cluster_tree.size() - 1; s >= 0 && clusters_to_analyze.size(); --s) {
        unsigned_int_type scale_id = s;
        std::vector<scalar_type> cluster_influence(clusters_to_analyze.size(), 0);
        scalar_type unclustered_influence(0);

        for (const auto& aoi : influence[scale_id]) {
          bool found = false;
          for (size_t i = 0; i < clusters_to_analyze.size(); ++i) {
            auto it = _cluster_tree[scale_id][clusters_to_analyze[i]].landmarks().find(aoi.first);
            if (it != _cluster_tree[scale_id][clusters_to_analyze[i]].landmarks().end()) {
              cluster_influence[i] += aoi.second;
              found = true;
            }
          }
          if (!found) {
            unclustered_influence += aoi.second;
          }
        }

        scalar_type max(unclustered_influence);
        int_type cluster_id(-1);
        for (size_t i = 0; i < clusters_to_analyze.size(); ++i) {
          if (cluster_influence[i] > max) {
            max = cluster_influence[i];
            cluster_id = _cluster_tree[scale_id][clusters_to_analyze[i]].id();
          }
        }

        if (cluster_id == -1) {
          return;
        }

        res = std::tuple<unsigned_int_type, int_type, scalar_type>(scale_id, cluster_id, max);
        //compute children nodes
        clusters_to_analyze.clear();
        if (s != 0) {
          for (size_t i = 0; i < _cluster_tree[s - 1].size(); ++i) {
            if (_cluster_tree[s - 1][i].parent_id() == cluster_id) {
              clusters_to_analyze.push_back(i);
            }
          }
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix, typename unsigned_integer, typename integer>
    void HierarchicalSNE<scalar, sparse_scalar_matrix, unsigned_integer, integer>::ClusterTree::computePointsToClusterAssociation(const HierarchicalSNE& hsne, std::vector<std::tuple<unsigned_int_type, int_type, scalar_type>>& res) {
      res.resize(hsne.scale(0).size());

      //#ifdef __USE_GCD__
      //      std::cout << "GCD dispatch, hierarchical_sne_inl 1227.\n";
      //      dispatch_apply(res.size(), dispatch_get_global_queue(0, 0), ^(size_t i) {
      //#else
#pragma omp parallel for
      for (int_type i = 0; i < res.size(); ++i) {
        //#endif //__USE_GCD__
        computePointToClusterAssociation(hsne, i, res[i]);
      }
      //#ifdef __USE_GCD__
      //      );
      //#endif

    }


    ///////////////////////////////////////////////////////////////////////////////////7

    namespace IO {

      template <typename hsne_type, class output_stream_type>
      void saveHSNE(const hsne_type& hsne, output_stream_type& stream, utils::AbstractLog* log) {
        checkAndThrowLogic(hsne.hierarchy().size(), "Cannot save an empty H-SNE hierarchy!!!");

        utils::secureLog(log, "Saving H-SNE hierarchy to file");

        //Version
        float major_version = 0.f;
        float minor_version = 1.f;
        stream.write(reinterpret_cast<char*>(&major_version), sizeof(float));
        stream.write(reinterpret_cast<char*>(&minor_version), sizeof(float));

        utils::secureLog(log, "Saving with version marker " + std::to_string(static_cast<int>(major_version)) + "." + std::to_string(static_cast<int>(minor_version)));

        //Number of scales
        size_t num_scales = hsne.hierarchy().size();
        stream.write(reinterpret_cast<char*>(&num_scales), sizeof(size_t));
        {
          //The first scale contains only the transition matrix
          const auto& scale = hsne.scale(0);
          size_t n = scale.size();

          utils::secureLogValue(log, "Saving scale", 0);
          utils::secureLog(log, "\tsize", n);
          stream.write(reinterpret_cast<char*>(&n), sizeof(size_t));
          utils::secureLog(log, "\t... transition matrix ...");
          data::IO::saveSparseMatrix(scale._transition_matrix, stream, log);
        }

        for (size_t s = 1; s < num_scales; ++s) {
          const auto& scale = hsne.scale(s);
          size_t n = scale.size();

          utils::secureLogValue(log, "Saving scale", s);
          utils::secureLogValue(log, "\tsize", n);
          stream.write(reinterpret_cast<char*>(&n), sizeof(size_t));
          utils::secureLog(log, "\t... transition matrix ...");
          data::IO::saveSparseMatrix(scale._transition_matrix, stream, log);
          utils::secureLog(log, "\t... landmarks to original data ...");
          data::IO::saveVector(scale._landmark_to_original_data_idx, stream, log);
          utils::secureLog(log, "\t... landmarks to previous scale ...");
          data::IO::saveVector(scale._landmark_to_previous_scale_idx, stream, log);
          utils::secureLog(log, "\t... landmark weights ...");
          data::IO::saveVector(scale._landmark_weight, stream, log);
          utils::secureLog(log, "\t... previous scale to current scale landmarks ...");
          data::IO::saveVector(scale._previous_scale_to_landmark_idx, stream, log);
          utils::secureLog(log, "\t... area of influence ...");
          data::IO::saveSparseMatrix(scale._area_of_influence, stream, log);
        }
      }

      ///////////////////////////////////////////////////////

      template <typename hsne_type, class input_stream_type>
      void loadHSNE(hsne_type& hsne, input_stream_type& stream, utils::AbstractLog* log) {
        utils::secureLog(log, "Loading H-SNE hierarchy from file");

        //Version
        float major_version = -1.f;
        float minor_version = -1.f;
        stream.read(reinterpret_cast<char*>(&major_version), sizeof(float));
        stream.read(reinterpret_cast<char*>(&minor_version), sizeof(float));

        utils::secureLog(log, "Loading file with version marker " + std::to_string(static_cast<int>(major_version)) + "." + std::to_string(static_cast<int>(minor_version)));

        checkAndThrowRuntime(major_version == 0.f, "Invalid major version");
        checkAndThrowRuntime(minor_version >= 0 && minor_version <= 1.f, "Invalid minor version");

        // backwards compatible loading
        if (major_version == 0.f && minor_version == 0.f)
        {
          typedef float io_unsigned_int_type;

          utils::secureLog(log, "Loading backwards compatible pre HDILib 1.3.0 file");

          //Number of scales
          io_unsigned_int_type num_scales = {};
          stream.read(reinterpret_cast<char*>(&num_scales), sizeof(io_unsigned_int_type));
          checkAndThrowRuntime(num_scales > 0, "Cannot load an empty hierarchy");
          {
            hsne.hierarchy().clear();
            hsne.hierarchy().push_back(typename hsne_type::Scale());
            auto& scale = hsne.scale(0);

            io_unsigned_int_type n = static_cast<io_unsigned_int_type>(scale.size());

            utils::secureLogValue(log, "Loading scale", 0);
            stream.read(reinterpret_cast<char*>(&n), sizeof(io_unsigned_int_type));
            utils::secureLog(log, "\tsize", n);
            utils::secureLog(log, "\t... transition matrix ...");
            data::IO::loadSparseScalarMatrix(scale._transition_matrix, stream, log);

            utils::secureLog(log, "\t... (init) landmarks to original data ...");
            scale._landmark_to_original_data_idx.resize(n);
            std::iota(scale._landmark_to_original_data_idx.begin(), scale._landmark_to_original_data_idx.end(), 0);
            utils::secureLog(log, "\t... (init) landmarks to previous scale ...");
            scale._landmark_to_previous_scale_idx.resize(n);
            std::iota(scale._landmark_to_previous_scale_idx.begin(), scale._landmark_to_previous_scale_idx.end(), 0);
            utils::secureLog(log, "\t... (init) landmark weights ...");
            scale._landmark_weight.resize(n, 1);
          }

          for (int s = 1; s < num_scales; ++s) {
            hsne.hierarchy().push_back(typename hsne_type::Scale());
            auto& scale = hsne.scale(s);
            io_unsigned_int_type n = {};

            utils::secureLogValue(log, "Loading scale", s);
            stream.read(reinterpret_cast<char*>(&n), sizeof(io_unsigned_int_type));
            utils::secureLogValue(log, "\tsize", n);
            utils::secureLog(log, "\t... transition matrix ...");
            data::IO::loadSparseScalarMatrix(scale._transition_matrix, stream, log);
            utils::secureLog(log, "\t... landmarks to original data ...");
            data::IO::loadUIntVector(scale._landmark_to_original_data_idx, stream, log);
            utils::secureLog(log, "\t... landmarks to previous scale ...");
            data::IO::loadUIntVector(scale._landmark_to_previous_scale_idx, stream, log);
            utils::secureLog(log, "\t... landmark weights ...");
            data::IO::loadScalarVector(scale._landmark_weight, stream, log);
            utils::secureLog(log, "\t... previous scale to current scale landmarks ...");
            data::IO::loadIntVector(scale._previous_scale_to_landmark_idx, stream, log);
            utils::secureLog(log, "\t... area of influence ...");
            data::IO::loadSparseScalarMatrix(scale._area_of_influence, stream, log);
          }
        }
        else
        {
          //Number of scales
          size_t num_scales = 0;
          stream.read(reinterpret_cast<char*>(&num_scales), sizeof(size_t));
          checkAndThrowRuntime(num_scales > 0, "Cannot load an empty hierarchy");
          {
            hsne.hierarchy().clear();
            hsne.hierarchy().push_back(typename hsne_type::Scale());
            auto& scale = hsne.scale(0);

            size_t n = scale.size();

            utils::secureLogValue(log, "Loading scale", 0);
            stream.read(reinterpret_cast<char*>(&n), sizeof(size_t));
            utils::secureLog(log, "\tsize", n);
            utils::secureLog(log, "\t... transition matrix ...");
            data::IO::loadSparseMatrix(scale._transition_matrix, stream, log);

            utils::secureLog(log, "\t... (init) landmarks to original data ...");
            scale._landmark_to_original_data_idx.resize(n);
            std::iota(scale._landmark_to_original_data_idx.begin(), scale._landmark_to_original_data_idx.end(), 0);
            utils::secureLog(log, "\t... (init) landmarks to previous scale ...");
            scale._landmark_to_previous_scale_idx.resize(n);
            std::iota(scale._landmark_to_previous_scale_idx.begin(), scale._landmark_to_previous_scale_idx.end(), 0);
            utils::secureLog(log, "\t... (init) landmark weights ...");
            scale._landmark_weight.resize(n, 1);

          }

          for (size_t s = 1; s < num_scales; ++s) {
            hsne.hierarchy().push_back(typename hsne_type::Scale());
            auto& scale = hsne.scale(s);
            size_t n = 0;

            utils::secureLogValue(log, "Loading scale", s);
            stream.read(reinterpret_cast<char*>(&n), sizeof(size_t));
            utils::secureLogValue(log, "\tsize", n);
            utils::secureLog(log, "\t... transition matrix ...");
            data::IO::loadSparseMatrix(scale._transition_matrix, stream, log);
            utils::secureLog(log, "\t... landmarks to original data ...");
            data::IO::loadVector(scale._landmark_to_original_data_idx, stream, log);
            utils::secureLog(log, "\t... landmarks to previous scale ...");
            data::IO::loadVector(scale._landmark_to_previous_scale_idx, stream, log);
            utils::secureLog(log, "\t... landmark weights ...");
            data::IO::loadVector(scale._landmark_weight, stream, log);
            utils::secureLog(log, "\t... previous scale to current scale landmarks ...");
            data::IO::loadVector(scale._previous_scale_to_landmark_idx, stream, log);
            utils::secureLog(log, "\t... area of influence ...");
            data::IO::loadSparseMatrix(scale._area_of_influence, stream, log);
          }

        }


      } // loadHSNE

    } // IO

  } // dr
} // hdi

#endif 
