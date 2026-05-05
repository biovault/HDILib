#include <hdi/data/embedding.h>
#include <hdi/dimensionality_reduction/hd_joint_probability_generator.h>
#include <hdi/dimensionality_reduction/knn_utils.h>
#include <hdi/dimensionality_reduction/gradient_descent_tsne_texture.h>
#include <hdi/dimensionality_reduction/tsne_parameters.h>
#include <hdi/utils/cout_log.h>

#include "offscreenWindow.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
    constexpr int num_points = 10000;
    constexpr int num_dims = 100;
    std::vector<float> data(num_points * num_dims);

    {
        std::cout << "Create random data\n";
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::generate(data.begin(), data.end(), [&]() {
            return dist(rng);
            });
    }

	try {
		hdi::dr::HDJointProbabilityGenerator<float>::Parameters prob_gen_param;
        prob_gen_param._perplexity = 30;
        prob_gen_param._aknn_algorithm = hdi::dr::KNN_FLANN; // choose Flann
        
        hdi::utils::CoutLog logger;

        hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type distributions;
		{
            std::cout << "Compute probability distributions\n";
            hdi::dr::HDJointProbabilityGenerator<float> prob_gen;
			prob_gen.setLogger(&logger);

			prob_gen.computeProbabilityDistributions(
				data.data(),
				num_dims,
				num_points,
				distributions,
				prob_gen_param);
			std::cout << "Probability distributions created\n";
		}
        
		{
			std::cout << "Create a Gradient Descent TSNE instance\n";
			hdi::dr::GradientDescentTSNETexture tSNE;
			tSNE.setLogger(&logger);

			auto offscreenBuffer = std::make_unique<OffscreenBufferGLFW>();
			offscreenBuffer->initialize();
			offscreenBuffer->bindContext();

			auto grad_desc_params = hdi::dr::TsneParameters();
			hdi::data::Embedding<float> embedding;

			std::cout << "Run TSNE gradient descent\n";
			tSNE.initialize(distributions, &embedding, grad_desc_params);

			constexpr uint32_t num_iterations = 1000;
			for (uint32_t it = 0; it < num_iterations; it++)
				tSNE.doAnIteration();

			offscreenBuffer->releaseContext();
			offscreenBuffer->destroyContext();
		}

        std::cout << "Finished\n";
        std::cout << std::flush;
    }
    catch (std::logic_error &e) {
        std::cout << "Logic error: " << e.what() << "\n";
        return 1;
    }
    catch (std::exception &e) {
        std::cout << "Exception: " << e.what() << "\n";
        return 1;        
    }
    catch (...) {     
        std::cout << "Unexpected exception\n"; 
        return 1;
    }

    return 0;
}