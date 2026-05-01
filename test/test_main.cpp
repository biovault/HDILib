#include "test_utils.h"

#include "hdi/dimensionality_reduction/knn_utils.h"

#include <catch2/catch_timer.hpp>

#include <cstdint>
#include <format>
#include <span>
#include <thread>
#include <vector>

using namespace testing;

/// /////////// ///
/// Definitions ///
/// //////////  ///

using int_type			= int32_t;
using unsigned_int_type = uint32_t;
using scalar_type		= float;

/// /////////// ///
///    Tests    ///
/// //////////  ///

TEST_CASE("Approximate knn") {
	info("TEST: Approximate KNN");

	using namespace hdi::dr;

	DataGenerator<scalar_type> gen;

	constexpr size_t numPoints = 10'000;
	constexpr size_t numDim = 16;

	std::vector<scalar_type> data = gen.randomMatrix(numDim, numPoints);

	// Shared settings
	KnnStatistics knnStatistics;

	auto makeKnnParameters = [](unsigned_int_type nn) -> KnnParameters {
		KnnParameters knnParams;
		unsigned_int_type numNeighbors = nn;
		knnParams._perplexity_multiplier = 3;
		knnParams._perplexity = (numNeighbors - 1) / 3.f;
		knnParams._num_trees = 8;
		knnParams._num_checks = 512;
		knnParams._aknn_algorithmP1 = 16;
		knnParams._aknn_algorithmP2 = 200;
		knnParams._aknn_metric = knn_distance_metric::KNN_METRIC_EUCLIDEAN;
		return knnParams;
		};

	// Reference knn
	std::vector<scalar_type> distances_squared_exact;
	std::vector<int_type> indices_exact;

	auto check_knn = [&knnStatistics, &data, &distances_squared_exact, &indices_exact](
		KnnParameters& knnParams, const knn_library knnlib, const std::string& info_text, auto check_fn
		) {
			Catch::Timer timer;

			knnParams._aknn_algorithm = knnlib;

			std::vector<scalar_type> distances_squared;
			std::vector<int_type> indices;

			info(info_text);
			timer.start();
			computeApproximateNearestNeighbors(data.data(), numDim, numPoints, knnParams, distances_squared, indices, knnStatistics, nullptr);
			printDuration(timer.getElapsedMicroseconds());

			check_fn(distances_squared_exact, distances_squared, indices_exact, indices);
		};

	SECTION("Exact equality for low number of neighbors") {
		info("SECTION: Parallel kNN -> Exact equality for low number of neighbors");
		unsigned_int_type numNeighbors = 15;
		KnnParameters knnParams = makeKnnParameters(numNeighbors);

		auto check_equality = [&numPoints, &numNeighbors](const std::vector<scalar_type>& D_exact, const std::vector<scalar_type>& D_test, const std::vector<int_type>& I_exact, const std::vector<int_type>& I_test) {
			const size_t flat_length = numPoints * numNeighbors;

			REQUIRE(I_exact.size() == flat_length);
			REQUIRE(I_test.size() == flat_length);
			REQUIRE(D_exact.size() == flat_length);
			REQUIRE(D_test.size() == flat_length);

			for (size_t i = 0; i < numPoints; i++) {
				std::span<const int_type> I_exact_n(I_exact.data() + i * numNeighbors, numNeighbors);
				std::span<const int_type> I_test_n(I_test.data() + i * numNeighbors, numNeighbors);
				std::span<const scalar_type> D_exact_n(D_exact.data() + i * numNeighbors, numNeighbors);
				std::span<const scalar_type> D_test_n(D_test.data() + i * numNeighbors, numNeighbors);

				for (size_t j = 0; j < numNeighbors; j++) {
					const bool I_same = I_exact_n[j] == I_test_n[j];
					const bool D_same = nearlyEqual(D_exact_n[j], D_test_n[j]);

					if (!I_same || !D_same) {
						print(std::pair{ I_exact_n[j], I_test_n[j] });
						print(std::pair{ D_exact_n[j], D_test_n[j] });
						print(I_exact_n);
						print(I_test_n);
						print(D_exact_n);
						print(D_test_n);
						break;
					}

					REQUIRE(I_same);
					REQUIRE(D_same);
				}
			}

			};

		// Exact
		info("  Computing exact...");
		exact_knn(data, numDim, numNeighbors, distances_squared_exact, indices_exact);

		// Approximate
		info("  Computing approximate...");
		check_knn(knnParams, knn_library::KNN_HNSW, "HNSW", check_equality);
		check_knn(knnParams, knn_library::KNN_ANNOY, "ANNOY", check_equality);
		check_knn(knnParams, knn_library::KNN_FLANN, "FLANN", check_equality);

		info("Section FINISHED");
	}

	SECTION("Recall for larger number of neighbors") {
		info("SECTION: Parallel kNN -> Recall for larger number of neighbors");

		unsigned_int_type numNeighbors = 100;
		KnnParameters knnParams = makeKnnParameters(numNeighbors);

		auto check_recall = [&numPoints, &numNeighbors](const std::vector<scalar_type>& D_exact, const std::vector<scalar_type>& D_test, const std::vector<int_type>& I_exact, const std::vector<int_type>& I_test) {
			size_t correct = 0;

			for (size_t i = 0; i < numPoints; i++) {
				std::span<const int_type> I_exact_n(I_exact.data() + i * numNeighbors, numNeighbors);
				std::span<const int_type> I_test_n(I_test.data() + i * numNeighbors, numNeighbors);
				std::span<const scalar_type> D_exact_n(D_exact.data() + i * numNeighbors, numNeighbors);
				std::span<const scalar_type> D_test_n(D_test.data() + i * numNeighbors, numNeighbors);

				for (size_t j = 0; j < numNeighbors; j++) {
					const bool I_same = I_exact_n[j] == I_test_n[j];
					const bool D_same = nearlyEqual(D_exact_n[j], D_test_n[j]);

					if (I_exact_n[j] == I_test_n[j] &&
						nearlyEqual(D_exact_n[j], D_test_n[j])) {
						correct++;
					}

				}
			}

			const auto recall = correct / static_cast<double>(numPoints * numNeighbors);
			info(std::format("Recall: {}", recall));

			};

		// Exact
		info("  Computing exact...");
		exact_knn(data, numDim, numNeighbors, distances_squared_exact, indices_exact);

		// Approximate
		info("  Computing approximate...");
		check_knn(knnParams, knn_library::KNN_HNSW, "HNSW", check_recall);
		check_knn(knnParams, knn_library::KNN_ANNOY, "ANNOY", check_recall);
		check_knn(knnParams, knn_library::KNN_FLANN, "FLANN", check_recall);

		info("Section FINISHED");
	}

}
