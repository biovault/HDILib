#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <filesystem>
#include <argparse/argparse.hpp>
#include <rapidcsv.h>

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
        .default_value(1000)
        .scan<'i', int>()
        .help("Number of iterations for tSNE (default: 1000).");
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    return 0;
}