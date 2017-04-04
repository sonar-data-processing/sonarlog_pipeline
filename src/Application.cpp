#include <opencv2/opencv.hpp>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "sonar_util/Converter.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonarlog_pipeline/Application.hpp"
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/QualityMetrics.hpp"

using namespace sonar_processing;

namespace sonarlog_pipeline {

Application *Application::instance_ = NULL;

Application*  Application::instance() {
    if (!instance_){
        instance_ = new Application();
    }
    return instance_;
}

void Application::init(const std::string& filename, const std::string& stream_name) {
    reader_.reset(new rock_util::LogReader(filename));
    plot_.reset(new base::Plot());
    stream_ = reader_->stream(stream_name);
}

void Application::process_next_sample() {
    base::samples::Sonar sample;
    stream_.next<base::samples::Sonar>(sample);

    /* current frame */
    std::vector<float> bearings = rock_util::Utilities::get_radians(sample.bearings);
    float angle = bearings[bearings.size()-1];
    uint32_t frame_height = 400;
    uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);

    /* roi frame */
    cv::Mat src(sample.beam_count, sample.bin_count, CV_32F, (void*) sample.bins.data());
    src.convertTo(src, CV_8U, 255);

    /* image enhancement */
    cv::Mat enhanced;
    preprocessing::adaptive_clahe(src, enhanced);

    // /* denoising process */
    // cv::Mat denoised;
    // rls.sliding_window(enhanced, denoised);
    //
    // /* convert to cartesian plane */
    // src.convertTo(src, CV_32F, 1.0 / 255.0);
    // sample.bins.assign((float*) src.datastart, (float*) src.dataend);
    // cv::Mat out1 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
    // sample.bins.assign((float*) denoised.datastart, (float*) denoised.dataend);
    // cv::Mat out2 = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
    //
    // cv::Mat out;
    // cv::hconcat(out1, out2, out);
    // cv::imshow("out", out);

    cv::waitKey(10);
}

void Application::process_logfile() {
    rls.setWindow_size(4);
    stream_.reset();
    while (stream_.current_sample_index() < stream_.total_samples()) process_next_sample();
    cv::waitKey();
}

void Application::plot(cv::Mat mat) {
    (*plot_)(image_util::mat2vector<float>(mat));
}

}
