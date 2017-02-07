#define BOOST_TEST_MODULE test_ArgumentParser
#include <boost/test/unit_test.hpp>
#include <cstdio>

#include "sonarlog_pipeline/ArgumentParser.hpp"
#include "base/test_config.h"

using namespace sonarlog_pipeline;

BOOST_AUTO_TEST_CASE(input_file_is_nonexistent)
{
    int argc = 3;
    char const *argv[3] = {
        "sonarlog_pipeline",
        "--input-file=nonexistent",
        "--stream-name=gemini.sonar_samples"
    };

    ArgumentParser argument_parser;
    BOOST_CHECK_MESSAGE(argument_parser.run(argc, argv) == false, "Return false if the input-file is nonexistent");
}

BOOST_AUTO_TEST_CASE(input_file_is_existent)
{
    char input_file_arg[256];
    int n = snprintf(input_file_arg, 256, "--input-file=%s/logs/gemini-ferry.0.log", DATA_PATH);

    BOOST_ASSERT(n >= 0 && n < 256);

    int argc = 3;
    char const *argv[3] = {
        "sonarlog_pipeline",
        input_file_arg,
        "--stream-name=gemini.sonar_samples"
    };

    ArgumentParser argument_parser;
    BOOST_CHECK_MESSAGE(argument_parser.run(argc, argv) == true, "Return false if the input-file is existent");
}
