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

#include <stdexcept>

namespace hdi {
  namespace dr {

    std::map<std::string, int> supported_knn_libraries()
    {
      std::map<std::string, int> result;
      result["FLANN"] = hdi::dr::KNN_FLANN;
      result["HNSW"] = hdi::dr::KNN_HNSW;
      result["ANNOY"] = hdi::dr::KNN_ANNOY;
      return result;
    }

    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib)
    {
      std::map<std::string, int> result;
      result["Euclidean"] = hdi::dr::KNN_METRIC_EUCLIDEAN;

      switch (knn_lib)
      {
      case hdi::dr::KNN_FLANN: {
        break;
      }
      case hdi::dr::KNN_HNSW: {
        result["Inner Product"] = hdi::dr::KNN_METRIC_INNER_PRODUCT;
        break;
      }
      case hdi::dr::KNN_ANNOY: {
        result["Cosine"] = hdi::dr::KNN_METRIC_COSINE;
        result["Manhattan"] = hdi::dr::KNN_METRIC_MANHATTAN;
        result["Dot"] = hdi::dr::KNN_METRIC_DOT;
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

      if (knnParameters._aknn_algorithm == hdi::dr::KNN_FLANN)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with Flann...");
        flann::Matrix<float> dataset(high_dimensional_data, num_dps, num_dim);
        flann::Matrix<float> query(high_dimensional_data, num_dps, num_dim);

        flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(knnParameters._num_trees));

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
      else if (knnParameters._aknn_algorithm == hdi::dr::KNN_HNSW)
      {
        hdi::utils::secureLog(_logger, "Computing approximated knn with HNSWLIB...");

        hnswlib::SpaceInterface<float>* space = NULL;
        switch (knnParameters._aknn_metric) {
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

        hnswlib::HierarchicalNSW<float> appr_alg(space, num_dps, knnParameters._aknn_algorithmP1, knnParameters._aknn_algorithmP2, 0);
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._trees_construction_time);
          utils::secureLog(_logger, "\tBuilding the search structure...");
          appr_alg.addPoint((void*)high_dimensional_data, (std::size_t)0);
          unsigned num_threads = std::thread::hardware_concurrency();
          hnswlib::ParallelFor(1, num_dps, num_threads, [&](size_t i, size_t threadId) {
            appr_alg.addPoint((void*)(high_dimensional_data + (i * num_dim)), (hnswlib::labeltype)i);
            });
        }
        distances_squared.resize(num_dps * nn);
        indices.resize(num_dps * nn);
        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._aknn_time);
          utils::secureLog(_logger, "\tAKNN queries...");
#pragma omp parallel for
          for (int i = 0; i < num_dps; ++i)
          {
            auto top_candidates = appr_alg.searchKnn(high_dimensional_data + (i * num_dim), (hnswlib::labeltype)nn);
            while (top_candidates.size() > nn) {
              top_candidates.pop();
            }
            auto* distances_offset = distances_squared.data() + (i * nn);
            auto indices_offset = indices.data() + (i * nn);
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
      }
      else // (knnParameters._aknn_algorithm == hdi::dr::KNN_ANNOY)
      {
        using namespace Annoy;
        hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy...");

        int search_k = nn * knnParameters._num_trees;

        distances_squared.resize(num_dps * nn);
        indices.resize(num_dps * nn);

        AnnoyIndexInterface<int32_t, double>* tree = nullptr;
        switch (knnParameters._aknn_metric) {
        case hdi::dr::KNN_METRIC_EUCLIDEAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int32_t, double, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_COSINE:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Cosine distances ...");
          tree = new AnnoyIndex<int32_t, double, Angular, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        case hdi::dr::KNN_METRIC_MANHATTAN:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Manhattan distances ...");
          tree = new AnnoyIndex<int32_t, double, Manhattan, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
          //case hdi::dr::KNN_METRIC_HAMMING:
          //  hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          //  tree = new AnnoyIndex<int32_t, double, Hamming, Kiss64Random>(num_dim);
          //  break;
        case hdi::dr::KNN_METRIC_DOT:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Dot product distances ...");
          tree = new AnnoyIndex<int32_t, double, DotProduct, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        default:
          hdi::utils::secureLog(_logger, "Computing approximated knn with Annoy using Euclidean distances ...");
          tree = new AnnoyIndex<int32_t, double, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy>(num_dim);
          break;
        }

        {
          utils::ScopedTimer<float, utils::Seconds> timer(knnStatistics._trees_construction_time);
          utils::secureLog(_logger, "\tBuilding the search structure...");

          for (unsigned int i = 0; i < num_dps; ++i) {
            double* vec = new double[num_dim];
            for (unsigned int z = 0; z < num_dim; ++z) {
              vec[z] = high_dimensional_data[i * num_dim + z];
            }
            tree->add_item(i, vec);
          }
          tree->build(knnParameters._num_trees);

          // Sample check if it returns enough neighbors
          std::vector<int> closest;
          std::vector<double> closest_distances;
          for (int n = 0; n < 100; n++) {
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);
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
            std::vector<double> closest_distances;
            tree->get_nns_by_item(n, nn, search_k, &closest, &closest_distances);

            // Copy current row
            for (unsigned int m = 0; m < nn; m++) {
              indices[n * nn + m] = closest[m];
              distances_squared[n * nn + m] = closest_distances[m] * closest_distances[m];
            }
          }
        }
        delete tree;
      }
    }

  } // dr
} //hdi
