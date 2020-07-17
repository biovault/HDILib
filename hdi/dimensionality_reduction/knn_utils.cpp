#include "knn_utils.h"
#include <stdexcept>

#ifdef HNSWLIB_FOUND
#ifdef _MSC_VER
#if(_MSC_VER >= 1900)
#define HNSWLIB_SUPPORTED
#endif //(_MSC_VER >= 1900)
#else // _MSC_VER
#if (__cplusplus >=201103)
#define HNSWLIB_SUPPORTED
#endif //(__cplusplus >=201103)
#endif // _MSC_VER
#endif //HNSWLIB_FOUND

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
      result["FLANN"] = hdi::dr::KNN_FLANN;
#ifdef HNSWLIB_SUPPORTED
      result["HNSW"] = hdi::dr::KNN_HNSW;
#endif
#ifdef __USE_ANNOY__
      result["ANNOY"] = hdi::dr::KNN_ANNOY;
#endif
      return result;
    }

    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib)
    {
      std::map<std::string, int> result;
      result["Euclidean"] = hdi::dr::KNN_METRIC_EUCLIDEAN;

      switch (knn_lib)
      {
      case hdi::dr::KNN_FLANN: {
        return result;
      }
      case hdi::dr::KNN_HNSW: {
        result["Inner Product"] = hdi::dr::KNN_METRIC_INNER_PRODUCT;
        return result;
      }
      case hdi::dr::KNN_ANNOY: {
        result["Cosine"] = hdi::dr::KNN_METRIC_COSINE;
        result["Manhattan"] = hdi::dr::KNN_METRIC_MANHATTAN;
        result["Dot"] = hdi::dr::KNN_METRIC_DOT;
        return result;
      }

      default: {
        throw std::out_of_range("knn_lib value out of range");
      }
      }
    }
  }
}



