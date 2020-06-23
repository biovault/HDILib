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

    std::map<std::string, int> supported_knn_libraries()
    {
      std::map<std::string, int> result;
      result["FLANN"] = hdi::utils::KNN_FLANN;
#ifdef HNSWLIB_SUPPORTED
      result["HNSW"] = hdi::utils::KNN_HNSW;
#endif
#ifdef __USE_ANNOY__
      result["ANNOY"] = hdi::utils::KNN_ANNOY;
#endif
      return result;
    }

    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib)
    {
      std::map<std::string, int> result;
      result["Euclidean"] = hdi::utils::KNN_METRIC_EUCLIDEAN;

      switch (knn_lib)
      {
      case hdi::utils::KNN_FLANN: {
        return result;
      }
      case hdi::utils::KNN_HNSW: {
        result["Inner Product"] = hdi::utils::KNN_METRIC_INNER_PRODUCT;
        return result;
      }
      case hdi::utils::KNN_ANNOY: {
        result["Cosine"] = hdi::utils::KNN_METRIC_COSINE;
        result["Manhattan"] = hdi::utils::KNN_METRIC_MANHATTAN;
        result["Dot"] = hdi::utils::KNN_METRIC_DOT;
        return result;
      }

      default: {
        throw std::out_of_range("knn_lib value out of range");
      }
      }
    }
  }
}



