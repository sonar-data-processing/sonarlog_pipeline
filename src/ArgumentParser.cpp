#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "sonarlog_pipeline/ArgumentParser.hpp"

using namespace boost;

namespace sonarlog_pipeline {

ArgumentParser::ArgumentParser()
    : input_files_()
    , stream_name_("") {
}

ArgumentParser::~ArgumentParser() {
}

bool ArgumentParser::run(int argc, char const *argv[]) {
    std::string app_name = boost::filesystem::basename(argv[0]);

    program_options::options_description desc("create video from sonar scan log");

    desc.add_options()
        ("input-files,i", program_options::value<std::vector<std::string> >()->required(), "the input files path")
        ("stream-name,s", program_options::value<std::string>()->default_value("sonar.sonar_scan_samples"), "the stream name")
        ("help,h", "show the command line description");

    program_options::positional_options_description pd;

    pd.add("input-file", 1);

    program_options::variables_map vm;

    try {
        program_options::store(program_options::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return false;
        }

        if (vm.count("input-files")) {
            input_files_ = vm["input-files"].as<std::vector<std::string> >();

            for (size_t i = 0; i < input_files_.size(); i++) {
                if (!file_exists(input_files_[i])){
                    std::cerr << "ERROR: input-files not found" << std::endl;
                    return false;
                }
            }
        }

        stream_name_ = vm["stream-name"].as<std::string>();

        program_options::notify(vm);
    } catch (boost::program_options::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    return true;
}

bool ArgumentParser::file_exists(std::string filename) {

    if (!filesystem::exists(filename) &&
        !filesystem::exists(filesystem::path(filesystem::current_path()).string() + "/" + filename)){
        return false;
    }

    return true;
}

std::string ArgumentParser::get_filename(std::string file_path) {
    filesystem::path p(file_path);
    return p.stem().string();
}


} /* namespace sonarlog_pipeline */
