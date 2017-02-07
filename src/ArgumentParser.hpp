#ifndef ArgumentParser_hpp
#define ArgumentParser_hpp

#include <string>
#include <vector>

namespace sonarlog_pipeline {

class ArgumentParser {
public:

    ArgumentParser();
    virtual ~ArgumentParser();

    std::vector<std::string> input_files() const {
        return input_files_;
    }

    std::string app_name() const {
        return app_name_;
    }

    std::string stream_name() const {
        return stream_name_;
    }

    bool run(int argc, char const *argv[]);

private:

    bool file_exists(std::string filename);
    std::string get_filename(std::string file_path);

    std::vector<std::string> input_files_;
    std::string stream_name_;
    std::string app_name_;

};

} /* namespace sonarlog_pipeline */

#endif /* ArgumentParser_hpp */
