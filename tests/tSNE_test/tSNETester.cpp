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
#include <algorithm>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

void save_to_csv(std::vector<float> embedding, std::string output, int iter = -1) {
    auto outname = (iter == -1) ? output + ".csv" : output + "_" + std::to_string(iter) + ".csv";
    std::fstream s{ outname, s.out };
    for (auto it = embedding.begin(); it != embedding.end();) {
        float x = *it++;
        if (it == embedding.end()) break;
        float y = *it++;
        s << std::setprecision(6) << x << "," << std::setprecision(6) << y << "\n";
    }
    s.flush();
    s.close();
}

std::vector<float> perform_tSNE(unsigned int num_points, unsigned int num_dimensions, std::vector<float> data, std::string output, int stepsoutput, int iterations = 1000, int perplexity = 30, int exaggeration_iter = 250, hdi::dr::knn_library knn_algorithm = hdi::dr::knn_library::KNN_HNSW, hdi::dr::knn_distance_metric knn_distance_metric = hdi::dr::knn_distance_metric::KNN_METRIC_EUCLIDEAN, int num_target_dimensions = 2) {

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
    tSNE.setType(hdi::dr::GradientDescentTSNETexture::COMPUTE_SHADER_VULKAN);
    
    tSNE_param._embedding_dimensionality = num_target_dimensions;
    tSNE_param._mom_switching_iter = exaggeration_iter;
    tSNE_param._remove_exaggeration_iter = exaggeration_iter;
    tSNE_param._exaggeration_factor = 4.0;
    prob_gen_param._perplexity = perplexity;
    prob_gen_param._aknn_metric = knn_distance_metric;
    prob_gen_param._aknn_algorithm = knn_algorithm;
    std::cout << "calculate knn" << std::endl;
    prob_gen.computeJointProbabilityDistribution(
        data.data(),
        num_dimensions,
        num_points,
        distributions,
        prob_gen_param);

    std::cout << "knn complete" << std::endl;
    float gradient_desc_comp_time;
    { // timed scope
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
        tSNE.initializeWithJointProbabilityDistribution(distributions, &embedding, tSNE_param);
        try {
            for (int iter = 0; iter < iterations; ++iter) {
                tSNE.doAnIteration();
                std::cout << "Iter: " << iter << " kl_divergence: " << tSNE.kl_divergence << "\n";
                if (stepsoutput > 0) {
                    if (iter > 0 && iter % stepsoutput == 0) {
                        save_to_csv(embedding.getContainer(), output, iter);
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception with error: " << e.what() << std::endl;
            return std::vector<float>();
        }
    }
    std::cout << "Gradient descent (sec) " << gradient_desc_comp_time << "\n";
    std::cout << "... done!\n";
    return embedding.getContainer();
}

std::vector<float> perform_tSNE_OpenGL(
    unsigned int num_points, 
    unsigned int num_dimensions, 
    std::vector<float> data, 
    std::string output, 
    int stepsoutput, 
    int iterations = 1000, 
    int perplexity = 30, 
    bool raster = false, 
    int exaggeration_iter = 250, 
    hdi::dr::knn_library knn_algorithm = hdi::dr::knn_library::KNN_HNSW, 
    hdi::dr::knn_distance_metric knn_distance_metric = hdi::dr::knn_distance_metric::KNN_METRIC_EUCLIDEAN, 
    int num_target_dimensions = 2) {

    if (!glfwInit()) {
        throw std::runtime_error("Unable to initialize GLFW.");
    }
    GLFWwindow* offscreen_context;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // invisible - ie offscreen, window
    offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
    if (offscreen_context == NULL) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(offscreen_context);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize OpenGL context");
    }

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
    if (raster)
        tSNE.setType(hdi::dr::GradientDescentTSNETexture::RASTER);

    tSNE_param._embedding_dimensionality = num_target_dimensions;
    tSNE_param._mom_switching_iter = exaggeration_iter;
    tSNE_param._remove_exaggeration_iter = exaggeration_iter;
    tSNE_param._exaggeration_factor = 4.0;
    prob_gen_param._perplexity = perplexity;
    prob_gen_param._aknn_metric = knn_distance_metric;
    prob_gen_param._aknn_algorithm = knn_algorithm;
    std::cout << "calculate knn" << std::endl;
    prob_gen.computeJointProbabilityDistribution(
        data.data(),
        num_dimensions,
        num_points,
        distributions,
        prob_gen_param);

    std::cout << "knn complete" << std::endl;
    float gradient_desc_comp_time;
    { // timed scope
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
        tSNE.initializeWithJointProbabilityDistribution(distributions, &embedding, tSNE_param);
        try {
            for (int iter = 0; iter < iterations; ++iter) {
                tSNE.doAnIteration();
                std::cout << "Iter: " << iter;
                if (!raster)
                    std::cout << " kl_divergence: " << tSNE.kl_divergence;
                else 
                    std::cout << "\n";
                if (stepsoutput > 0) {
                    if (iter > 0 && iter % stepsoutput == 0) {
                        save_to_csv(embedding.getContainer(), output, iter);
                    }
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception with error: " << e.what() << std::endl;
            return std::vector<float>();
        }
    }
    std::cout << "Gradient descent (sec) " << gradient_desc_comp_time << "\n";
    std::cout << "... done!\n";
    return embedding.getContainer();
}

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec2d) {
    std::vector<T> result;
    for (const auto& v : vec2d)
        result.insert(result.end(), v.begin(), v.end());
    return result;
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
        .default_value(std::string("embedding"))
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
    bool binary = false;
    program.add_argument("-b", "--binary")
        .store_into(binary)
        .default_value(false)
        .help("The input file is binary not csv");
    bool opengl = false;
    program.add_argument("-g", "--opengl")
        .store_into(opengl)
        .default_value(false);
    bool raster = false;
    program.add_argument("-r", "--raster")
        .store_into(raster)
        .default_value(false)
        .help("Use RASTER iplementation instead of VULKAN");
    auto& algoGroup = program.add_mutually_exclusive_group();
    algoGroup.add_argument("--opengl");
    algoGroup.add_argument("--raster");
    program.add_argument("-d", "--dimension")
        .required()
        .scan<'i', unsigned int>()
        .help("Length of row - addument uchar values");
    program.add_argument("-n", "--numpoints")
        .default_value(-1)
        .scan<'i', int>()
        .help("Number of points to take from set (default -1 means all).");
    program.add_argument("-s", "--stepsoutput")
        .default_value(-1)
        .scan<'i', int>()
        .help("Number steps between output (default -1 is no intermediate output).");

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
    auto dim = program.get<unsigned int>("dimension");
    auto subset = program.get<int>("numpoints");
    auto stepsoutput = program.get<int>("stepsoutput");
    //auto loglevel = program.get<unsigned int>("loglevel");
    std::vector<float> data;
    unsigned int num_points;
    if (!binary) {
        //skip comment lines - use to comment out header
        rapidcsv::Document doc(
            csvpath.string(), 
            rapidcsv::LabelParams(0,0), 
            rapidcsv::SeparatorParams(),
            rapidcsv::ConverterParams(), 
            rapidcsv::LineReaderParams(true));

        std::vector<std::vector<float>> col_holder;
        for (auto i = 0; i < dim; ++i) {
            col_holder.push_back(doc.GetColumn<float>(static_cast<size_t>(i)));
        }

        num_points = col_holder[0].size();
        if (subset != -1) {
            num_points = subset;
        }
        std::cout << "Loading csv: " << num_points << " points" << std::endl;
        data = std::vector<float>(num_points * dim);

        for (auto i = 0; i < num_points; ++i) {
            for( auto j = 0; j < dim; ++j) {
                data[dim * i + j] = col_holder[j][i];
            }
        }
    }
    else {

        std::ifstream fileCheck(csvpath.string(), std::ios::binary);
        fileCheck.seekg(0, std::ios::end);
        std::streampos fileSize = fileCheck.tellg();
        fileCheck.close();
        std::cout << "subset num points: " << subset << "\n";
        auto all_points = fileSize / (dim * sizeof(float));
        if (subset != -1) {
            num_points = subset;
        }
        else {
            num_points = all_points;
        }
        // each point is a vector of float dimensions
        auto data2D = std::vector<std::vector<float>>();
        data2D.resize(all_points);
        
        std::cout << "Loading: " << num_points << " float binary points" << std::endl;
        std::ifstream file(csvpath.string(), std::ios::binary);
        auto count = size_t(0);
        auto point_count = 0;
        if (file) {
            float pnt;
            while (count < fileSize) {
                auto point_vec = std::vector<float>();
                point_vec.resize(dim);
                for (auto d = 0; d < dim; d++) {
                    file.read(reinterpret_cast<char*>(&pnt), sizeof(pnt));
                    point_vec[d] = pnt;
                    count += sizeof(float);
                }
                data2D[point_count] = point_vec;
                ++point_count;
            }
        }
        // if a subset is required sample it randomly
        if (all_points > num_points) {
            std::vector<std::vector<float>> sample;
            auto randdev = std::mt19937{}; //std::random_device{}()
            std::sample(
                data2D.begin(), 
                data2D.end(), 
                std::back_inserter(sample), 
                num_points, 
                randdev);  
            data2D = sample;
        }
        // flatten the end result
        data = flatten(data2D);
    }

    std::vector<float> embedding;
    if (opengl || raster)
        embedding = perform_tSNE_OpenGL(num_points, dim, data, output, stepsoutput, iterations, perplexity, raster);
    else
        embedding = perform_tSNE(num_points, dim, data, output, stepsoutput, iterations, perplexity);
    save_to_csv(embedding, output);

    return 0;
}

// Xmas tree small test
// -p 13 -i 500 -d 2 D:\Data\ML\xmas\data.csv
// Full MNIST
// -b -p 30 -i 1000 -n 70000 -d 784 D:\Data\ML\MNIST\MNIST_70000.bin
// Small MNIST
// -b -p 30 -i 1000 -n 1000 -d 784 D:\Data\ML\MNIST\MNIST_70000.bin