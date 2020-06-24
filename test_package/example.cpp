#include <hdi/dimensionality_reduction/tsne.h>
#include <hdi/data/embedding.h>
#include <hdi/dimensionality_reduction/hd_joint_probability_generator.h>
#include <hdi/dimensionality_reduction/knn_utils.h>
#include <hdi/dimensionality_reduction/gradient_descent_tsne_texture.h>
#include <hdi/utils/cout_log.h>
#include <hdi/utils/log_helper_functions.h>

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>

using namespace hdi::dr;
using namespace hdi::data;
using namespace hdi::utils;

float f()
{ 
    float res = static_cast<float>(std::rand())/RAND_MAX;
    return res;
}

int main(int argc, char** argv)
{
    try {
        std::cout << "Create and initialize an Embedding\n";
        Embedding<float>::scalar_vector_type v(1000000); 
        std::cout << "Data vector created\n";        
        std::generate(v.begin(), v.end(), f);
        std::cout << "Data vector filled\n";
        hdi::data::Embedding<float> embedding;       
        

		HDJointProbabilityGenerator<float>::Parameters prob_gen_param;        
        prob_gen_param._perplexity = 30;
        prob_gen_param._aknn_algorithm = KNN_FLANN; //choose Flann
        
        HDJointProbabilityGenerator<float> prob_gen;
		HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type distributions;
        std::cout << "Compute probability distributions\n";
        prob_gen.computeProbabilityDistributions(
            v.data(),
            100,
            10000,
            distributions,
            prob_gen_param);
        std::cout << "Probability distributions created\n";    
        
        std::cout << "Create a Gradient Descent TSNE instance\n";
        GradientDescentTSNETexture tSNE; 
        CoutLog log;
        tSNE.setLogger(&log);
        assert(tSNE.isInitialized() == false);
        std::cout << "TSNE Created - not initialized\n";
        // No init attempted as this required an OpenGL context
        
        return 0;
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
}