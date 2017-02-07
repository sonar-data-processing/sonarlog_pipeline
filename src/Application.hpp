#ifndef Application_hpp
#define Application_hpp

#include <iostream>
#include <string>
#include <deque>
#include "rock_util/LogReader.hpp"
// #include "sonar_processing/TargetTrack.hpp"
#include "sonar_processing/Denoising.hpp"
#include "base/Plot.hpp"

using namespace sonar_processing;

namespace sonarlog_pipeline {

class Application {
public:

    void init(const std::string& filename, const std::string& stream_name);

    void process_logfile();

    void process_next_sample();

    void plot(cv::Mat mat);

    base::Plot& plot() {
        return *(plot_.get());
    }

    static Application* instance();

private:

    Application() {}

    ~Application() {}

    std::auto_ptr<rock_util::LogReader> reader_;
    rock_util::LogStream stream_;
    denoising::RLS rls;

    static Application *instance_;
    std::auto_ptr<base::Plot> plot_;
};

} /* namespace sonarlog_pipeline */



#endif /* Application_hpp */
