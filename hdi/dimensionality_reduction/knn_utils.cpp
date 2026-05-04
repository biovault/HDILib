#include "knn_utils.h"

#include "hdi/utils/abstract_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"

#pragma warning(disable:4477)
#include <annoylib.h>
#include <kissrandom.h>
#pragma warning(default:4477)
#include <flann/flann.h>
#include <hnswlib/hnswlib.h>
#include <hnswlib/space_l2.h>

#include <memory>
#include <stdexcept>
#include <thread>

namespace hdi {
  namespace dr {

    std::map<std::string, int> supported_knn_libraries()
    {
      std::map<std::string, int> result;
      result["FLANN"] = KNN_FLANN;
      result["HNSW"] = KNN_HNSW;
      result["ANNOY"] = KNN_ANNOY;
      return result;
    }

    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib)
    {
      std::map<std::string, int> result;
      result["Euclidean"] = KNN_METRIC_EUCLIDEAN;

      switch (knn_lib)
      {
      case KNN_FLANN: {
        break;
      }
      case KNN_HNSW: {
        result["Inner Product"] = KNN_METRIC_INNER_PRODUCT;
        break;
      }
      case KNN_ANNOY: {
        result["Cosine"] = KNN_METRIC_COSINE;
        result["Manhattan"] = KNN_METRIC_MANHATTAN;
        result["Dot"] = KNN_METRIC_DOT;
        break;
      }

      default: {
        throw std::out_of_range("knn_lib value out of range");
      }
      }

      return result;
    }


    void computeApproximateNearestNeighbors(float* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, const KnnParameters& knnParameters, std::vector<float>& distances_squared, std::vector<int>& indices, KnnStatistics& knnStatistics, utils::AbstractLog* _logger) {

      const int nn = knnParameters._perplexity * knnParameters._perplexity_multiplier + 1;

      if (knnParameters._aknn_algorithm == KNN_FLANN)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with Flann...");
        flann::Matrix<float> dataset(high_dimensional_data, num_dps, num_dim);
        flann::Matrix<float> query(high_dimensional_data, num_dps, num_dim);

        flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(knnParameters._num_trees));

        distances_squared.resize(num_dps * nn);
        indices.resize(num_dps * nn);
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._trees_construction_time);
          utils::secureLog(_logger, "\tBuilding the trees...");
          index.buildIndex();
        }
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._aknn_time);
          flann::Matrix<int> indices_mat(indices.data(), query.rows, nn);
          flann::Matrix<float> dists_mat(distances_squared.data(), query.rows, nn);
          flann::SearchParams flann_params(knnParameters._num_checks);
          flann_params.cores = 0; // all cores
          utils::secureLog(_logger, "\tAKNN queries...");
          index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
        }
      }
      else if (knnParameters._aknn_algorithm == KNN_HNSW)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with HNSWLIB...");

        std::unique_ptr<hnswlib::SpaceInterface<float>> space;
        switch (knnParameters._aknn_metric) {
        case KNN_METRIC_EUCLIDEAN:      space = std::make_unique<hnswlib::L2Space>(num_dim); break;
        case KNN_METRIC_INNER_PRODUCT:  space = std::make_unique<hnswlib::InnerProductSpace>(num_dim); break;
        default:                        space = std::make_unique<hnswlib::L2Space>(num_dim); break;
        }

        hnswlib::HierarchicalNSW<float> index(space.get(), num_dps, knnParameters._aknn_algorithmP1, knnParameters._aknn_algorithmP2);
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._trees_construction_time);
          utils::secureLog(_logger, "\tBuilding the search structure...");
          index.addPoint((void*)high_dimensional_data, 0);
          const unsigned num_threads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
          for (int i = 1; i < num_dps; ++i) {
            index.addPoint((void*)(high_dimensional_data + (i * num_dim)), i);
          }
        }
        index.setEf(knnParameters._aknn_algorithmP2);
        distances_squared.resize(num_dps * nn);
        indices.resize(num_dps * nn);
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._aknn_time);
          utils::secureLog(_logger, "\tAKNN queries...");

#pragma omp parallel for
          for (int i = 0; i < num_dps; ++i)
          {
            auto top_candidates = index.searchKnn(high_dimensional_data + (i * num_dim), nn);
            while (top_candidates.size() > nn) {
              top_candidates.pop();
            }
            auto* distances_offset = distances_squared.data() + (i * nn);
            auto indices_offset = indices.data() + (i * nn);
            int j = 0;
            while (!top_candidates.empty()) {
              auto rez = top_candidates.top();
              distances_offset[nn - j - 1] = rez.first;
              indices_offset[nn - j - 1] = rez.second;
              top_candidates.pop();
              ++j;
            }
          }
        }
      }
      else // (knnParameters._aknn_algorithm == KNN_ANNOY)
      {
        using AnnoyThreadPolicy = Annoy::AnnoyIndexSingleThreadedBuildPolicy;
        using AnnoyRng = Annoy::Kiss64Random;
        hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy...");

        const int search_k = nn * knnParameters._num_trees;

        distances_squared.resize(num_dps * nn);
        indices.resize(num_dps * nn);

        std::unique_ptr<Annoy::AnnoyIndexInterface<int32_t, float>> index;
        switch (knnParameters._aknn_metric) {
        case KNN_METRIC_EUCLIDEAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          index = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::Euclidean, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          break;
        case KNN_METRIC_COSINE:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Cosine distances ...");
          index = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::Angular, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          break;
        case KNN_METRIC_MANHATTAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          index = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::Manhattan, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          break;
          //case KNN_METRIC_HAMMING:
          //  hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          //tree = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::Hamming, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          //  break;
        case KNN_METRIC_DOT:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Dot product distances ...");
          index = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::DotProduct, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          break;
        default:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          index = std::make_unique<Annoy::AnnoyIndex<int32_t, float, Annoy::Euclidean, AnnoyRng, AnnoyThreadPolicy>>(num_dim);
          break;
        }

        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._trees_construction_time);
          utils::secureLog(_logger, "\tBuilding the search structure...");

          for (unsigned int i = 0; i < num_dps; ++i) {
            const float* item = high_dimensional_data + static_cast<size_t>(i * num_dim);
            index->add_item(i, item);
          }
          index->build(knnParameters._num_trees);
        }
        
        {
          // Sample check if the index returns enough neighbors
          std::vector<int> closest;
          std::vector<float> closest_distances;
          for (int n = 0; n < 10; n++) {
            index->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);
            unsigned int neighbors_count = closest.size();
            if (neighbors_count < nn) {
              printf("Requesting %d neighbors, but ANNOY returned only %u. Please increase search_k\n", nn, neighbors_count);
              return;
            }
          }
        }

        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._aknn_time);
          hdi::utils::secureLog(_logger, "AKNN queries...");

#pragma omp parallel for
          for (int n = 0; n < num_dps; n++)
          {
            // Find nearest neighbors
            std::vector<int> closest;
            std::vector<float> closest_distances;
            closest.reserve(nn); // reserve to avoid reallocations 
            closest_distances.reserve(nn);
            index->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned int m = 0; m < nn; m++) {
              indices[n * nn + m] = closest[m];
              distances_squared[n * nn + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
      }
    }

  } // dr
} //hdi
