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

	//! Returns the number of supported KNN libraries. 
	//! This function should be considered deprecated and kept for backward comppatibility. New code should use the supported_knn_libraries function.
    int HierarchicalSNE_NrOfKnnAlgorithms();

    //! Returns both the name/label and index of the supported KNN libraries since this can depend on compiler support. 
	//! This function is especially useful for building user-interfaces where the user can select which KNN library to use for a specific task (e.g. t-SNE or HSNE). 
	//! Alternatively it can be used to offer a look-up table to translate the currently set KNN Library index back to human readable text.
    std::map<std::string, int> supported_knn_libraries();

    //! Returns the name/label and index of distance metrics supported by a specific KNN library.
	//! This function is especially useful for building user-interfaces where the user can select both a KNN library and a distance metric since not every KNN library supports the same distance metric. 
	//! Alternatively it can be used to offer a look-up table to translate the currently set KNN distance metric index back to human readable text.
    std::map<std::string, int> supported_knn_library_distance_metrics(int knn_lib);
  }
}
#endif // KNN_H
