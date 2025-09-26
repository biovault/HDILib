#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <filesystem>
#include <argparse/argparse.hpp>
#include <rapidcsv.h>
#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/data/embedding.h"
#include "hdi/data/panel_data.h"
#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/dimensionality_reduction/knn_utils.h"
#include "hdi/dimensionality_reduction/tsne.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"
#include <kompute/logger/Logger.hpp>


std::vector<float> perform_tSNE(unsigned int num_points, unsigned int num_dimensions, std::vector<float> data, int iterations = 1000, int perplexity = 30, int exaggeration_iter = 250, hdi::dr::knn_library knn_algorithm = hdi::dr::knn_library::KNN_HNSW, hdi::dr::knn_distance_metric knn_distance_metric = hdi::dr::knn_distance_metric::KNN_METRIC_EUCLIDEAN, int num_target_dimensions = 2) {

    hdi::dr::knn_library _knn_algorithm;
    hdi::dr::knn_distance_metric _knn_metric;
    hdi::dr::TsneParameters tSNE_param;
    using ProbGenType = hdi::dr::HDJointProbabilityGenerator<float>;
    ProbGenType::Parameters prob_gen_param;
    ProbGenType prob_gen;
    hdi::data::Embedding<float> embedding;
    using MapType = hdi::data::MapMemEff<uint32_t, float>;
    using SparseScalarMatrixType = std::vector<MapType>;
    SparseScalarMatrixType distributions;
    hdi::dr::GradientDescentTSNETexture tSNE;
    
    tSNE_param._embedding_dimensionality = num_target_dimensions;
    tSNE_param._mom_switching_iter = exaggeration_iter;
    tSNE_param._remove_exaggeration_iter = exaggeration_iter;
    prob_gen_param._perplexity = perplexity;
    prob_gen_param._aknn_metric = knn_distance_metric;
    prob_gen_param._aknn_algorithm = knn_algorithm;
    prob_gen.computeJointProbabilityDistribution(
        data.data(),
        num_dimensions,
        num_points,
        distributions,
        prob_gen_param);

    std::cout << "knn complete" << std::endl;
    tSNE.initializeWithJointProbabilityDistribution(distributions, &embedding, tSNE_param);
    try {
        for (int iter = 0; iter < iterations; ++iter) {
            tSNE.doAnIteration();
            std::cout << "Iter: " << iter << " kl_divergence: " << tSNE.kl_divergence << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception with error: " << e.what() << std::endl;
        return std::vector<float>();
    }
    std::cout << "... done!\n";
    return embedding.getContainer();
}

int main(int argc, const char** argv) {
    argparse::ArgumentParser program("tsne_tester", "<local-build>");

    program.add_description("A test program for tSNE functionality in HDIlib.");
    program.add_epilog("Intended for use in testing the HDILib. This is not a general purpose dimensionality reduction tool.");
    program.add_argument("csvfile")
        .required()
        .help("Path to the input data file (CSV format).")
        .action([](const std::string& value) { return std::filesystem::path(value); });
    program.add_argument("-o", "--output")
        .default_value(std::string("embedding.csv"))
        .help("Path to the output embedding file (CSV format).")
        .action([](const std::string& value) { return std::filesystem::path(value); });
    program.add_argument("-p", "--perplexity")
        .default_value(30.0)
        .scan<'g', double>()
        .help("Perplexity value for tSNE (default: 30.0).");
    program.add_argument("-i", "--iterations")
        .default_value(1000u)
        .scan<'i', unsigned int>()
        .help("Number of iterations for tSNE (default: 1000).");
    //program.add_argument("-l", "--loglevel")
    //    .default_value(2u)
    //    .scan<'l', unsigned int>()
    //    .help("0: Trace, 1: Debug, 2: Info, 3: Warn, 4: Error:, 5: Critical, 6: Off");
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto csvpath = program.get<std::filesystem::path>("csvfile");
    auto output = program.get<std::string>("output");
    auto perplexity = program.get<double>("perplexity");
    auto iterations = program.get<unsigned int>("iterations");
    //auto loglevel = program.get<unsigned int>("loglevel");

    rapidcsv::Document doc(csvpath.string());

    std::vector<float> X = doc.GetColumn<float>("X");
    std::vector<float> Y = doc.GetColumn<float>("Y");

    auto num_points = X.size();
    std::vector<float> data = std::vector<float>(num_points * 2);

    for (auto i = 0; i < num_points; ++i) {
        data[2 * i] = X[i];
        data[2 * i + 1] = Y[i];
    }

    auto embedding = perform_tSNE(num_points, 2, data, iterations, perplexity);
    std::fstream s{ output, s.out };
    for (auto it = embedding.begin(); it != embedding.end();) {
        float x = *it++;
        if (it == embedding.end()) break;
        float y = *it++;
        s << std::setprecision(6) << x << "," << std::setprecision(6) << y << "\n";
    }
    s.flush();
    s.close();

    return 0;
}