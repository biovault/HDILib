#include "knn_support.h"
#include "../utils/knn_utils.h"

namespace hdi {
  namespace dr {
    int HierarchicalSNE_NrOfKnnAlgorithms()
    {
      int numSupported = 1;
#ifdef HNSWLIB_SUPPORTED
      numSupported++;
#endif

#ifdef __USE_ANNOY__
      numSupported++;
#endif
      return numSupported;
    }

    std::vector<int> supported_knn_libraries()
    {
      std::vector<int> result(1, hdi::utils::KNN_FLANN);
#ifdef HNSWLIB_SUPPORTED
      result.push_back(hdi::utils::KNN_HNSW);
#endif
#ifdef __USE_ANNOY__
      result.push_back(hdi::utils::KNN_ANNOY);
#endif
      return result;
    }

    std::vector<int> supported_knn_library_distance_metrics(int knn_lib)
    {
      switch (knn_lib)
      {
      case hdi::utils::KNN_FLANN: {
        return std::vector<int>(1, hdi::utils::KNN_METRIC_EUCLIDEAN);
      }
      case hdi::utils::KNN_HNSW: {
        std::vector<int> result(2);
        result[0] = hdi::utils::KNN_METRIC_EUCLIDEAN;
        result[1] = hdi::utils::KNN_METRIC_INNER_PRODUCT;
        return result;

      }
      case hdi::utils::KNN_ANNOY: {
        std::vector<int> result(5);
        result[0] = hdi::utils::KNN_METRIC_EUCLIDEAN;
        result[1] = hdi::utils::KNN_METRIC_COSINE;
        result[2] = hdi::utils::KNN_METRIC_MANHATTAN;
        result[3] = hdi::utils::KNN_METRIC_DOT;
        return result;
      }

      default: {
        throw std::out_of_range("knn_lib value out of range");
      }
      }
    }
  }

}



