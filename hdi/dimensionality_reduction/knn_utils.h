/*
 *
 * Copyright (c) 2019, BIOVAULT (Leiden University Medical Center, Delft University of Technology)
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
 * THIS SOFTWARE IS PROVIDED BY BIOVAULT ''AS IS'' AND ANY EXPRESS
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

#ifndef KNN_H
#define KNN_H
#include <vector>
#include <map>
#include <string>

namespace hdi{
  namespace dr{
  
    enum knn_library
    {
      KNN_FLANN = -1,
      KNN_HNSW = 0,
      KNN_ANNOY = 1
    };
 
    enum knn_distance_metric
    {
      KNN_METRIC_EUCLIDEAN = 0,
      KNN_METRIC_COSINE = 1,
      KNN_METRIC_INNER_PRODUCT = 2,
      KNN_METRIC_MANHATTAN = 3,
      KNN_METRIC_HAMMING = 4,
      KNN_METRIC_DOT = 5
    };

    int HierarchicalSNE_NrOfKnnAlgorithms();

    //! returns which knn libraries are supported.
    std::map<std::string, int> supported_knn_libraries();

    //! returns a map with distance metrics supported for a specific knn library
    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib);
  }
}
#endif // KNN_H
