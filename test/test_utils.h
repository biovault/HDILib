#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <cmath>
#include <queue>
#include <random>
#include <string>
#include <span>
#include <vector>
#include <cstdint>
#include <utility>

namespace testing {

	/// /////// ///
	/// TESTING ///
	/// /////// ///

	// Test helper that compares two values with tolerance
	inline void expectNear(const float expected, const float actual, const float tolerance = 1e-5f,
		const std::string& test_name = "") {
		if (std::fabs(expected - actual) > tolerance) {
			std::cerr << "Test failed: " << test_name
				<< "\n  Expected: " << expected
				<< "\n  Actual: " << actual
				<< "\n  Diff: " << std::fabs(expected - actual)
				<< "\n  Tolerance: " << tolerance << std::endl;
		}
		CHECK(expected == Catch::Approx(actual).margin(tolerance));
	}

	inline bool nearlyEqual(const float a, const float b, const float epsilon = 1e-6) {
		return std::fabs(a - b) < epsilon;
	}

	template<typename scalar = float>
	scalar l2_squared(const scalar* a, const scalar* b, size_t dim) {
		scalar dist = 0.0f;
		for (size_t i = 0; i < dim; i++) {
			scalar diff = a[i] - b[i];
			dist += diff * diff;
		}
		return dist;
	}

	template<typename scalar = float, typename integer = int32_t>
	void exact_knn(
		const std::vector<scalar>& data,
		const size_t dim,
		const size_t k,
		std::vector<scalar>& all_distances,   // size n_points * k
		std::vector<integer>& all_indices)     // size n_points * k
	{
		const integer n_points = static_cast<integer>(data.size() / dim);
		all_distances.resize(n_points * k);
		all_indices.resize(n_points * k);

		for (integer q = 0; q < n_points; ++q) {
			const scalar* query_ptr = data.data() + q * dim;

			using Result = std::pair<scalar, integer>;
			std::priority_queue<Result> heap;

			for (integer i = 0; i < n_points; ++i) {
				const scalar dist = l2_squared(data.data() + i * dim, query_ptr, dim);
				if (static_cast<integer>(heap.size()) < k) {
					heap.emplace(dist, i);
				}
				else if (dist < heap.top().first) {
					heap.pop();
					heap.emplace(dist, i);
				}
			}

			scalar* d_row = all_distances.data() + q * k;
			integer* i_row = all_indices.data() + q * k;
			for (size_t i = heap.size(); i-- > 0;) {
				d_row[i] = heap.top().first;
				i_row[i] = heap.top().second;
				heap.pop();
			}
		}
	}

	/// ///////////// ///
	/// DATA CREATION ///
	/// ///////////// ///

	template<typename scalar = float>
	class DataGenerator {
	public:
		DataGenerator(const scalar rangeMin = -1.f, const scalar rangeMax = 1.f) {
			gen.seed(42);
			dist = std::uniform_real_distribution<scalar>(rangeMin, rangeMax);
		}

		// Helper function to generate random vector
		std::vector<scalar> randomVector(const size_t dim) {
			std::vector<scalar> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = dist(gen);
			}
			return vec;
		}

		// Helper function to generate random matrix
		std::vector<scalar> randomMatrix(const size_t dim, const size_t numPoints) {
			return randomVector(dim * numPoints);
		}

		// Helper function to generate constant vector
		std::vector<scalar> constantVector(const size_t dim, const scalar value) {
			return std::vector<scalar>(dim, value);
		}

		// Helper function to generate linearly increasing vector
		std::vector<scalar> linearVector(const size_t dim, const scalar start = 0.0f, const scalar step = 1.0f) {
			std::vector<scalar> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = start + i * step;
			}
			return vec;
		}

	private:
		std::mt19937 gen;
		std::uniform_real_distribution<scalar> dist;
	};

	/// /////// ///
	/// LOGGING ///
	/// /////// ///

	template<typename T, typename S>
	inline void print(const std::pair<T, S>& pair) {
		std::cout << pair.first << " : " << pair.second << "\n";
	}

	template<typename T, typename S>
	inline void print(const std::vector<std::pair<T, S>>& pairs) {
		for (const auto& pair : pairs)
			print(pair);
		std::cout << std::endl;
	}

	template<typename T>
	void print(const std::vector<T>& vec) {
		for (const auto& val : vec) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}

	template<typename T>
	void print(const std::span<T>& vec) {
		for (const auto& val : vec) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}

	inline void info(const std::string& message) {
		std::cout << message << std::endl;
	}

	inline void printDuration(const uint64_t t, const std::string& unit = "ms") {
		std::cout << "Duration: " << t <<  unit << std::endl;
	}

}

#endif // TEST_UTILS_H
