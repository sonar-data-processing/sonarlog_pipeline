#include <iostream>
#include "sonarlog_pipeline/ArgumentParser.hpp"
#include "sonarlog_pipeline/Application.hpp"

using namespace sonarlog_pipeline;

int main(int argc, char const *argv[]) {

    ArgumentParser argument_parser;
    if (argument_parser.run(argc, argv)) {
        std::cout << "Sonar's log processing" << std::endl;
        for (size_t i = 0; i < argument_parser.input_files().size(); i++) {
            std::cout << "intput-file: " << argument_parser.input_files()[i]  << std::endl;
            std::cout << "stream-name: " << argument_parser.stream_name() << "\n" << std::endl;
            Application::instance()->init(argument_parser.input_files()[i], argument_parser.stream_name());
            Application::instance()->process_logfile();
        }

    }

    return 0;
}
