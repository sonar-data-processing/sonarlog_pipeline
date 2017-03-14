#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Clustering.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_pipeline/Application.hpp"

using namespace sonarlog_pipeline;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

inline void load_sonar_holder(const base::samples::Sonar& sample, sonar_processing::SonarHolder& sonar_holder) {
    sonar_holder.Reset(sample.bins,
        rock_util::Utilities::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count);
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
        DATA_PATH_STRING + "/logs/pipeline-front.1.log",
        DATA_PATH_STRING + "/logs/pipeline-parallel.0.log"
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(3);
    sonar_processing::SonarHolder sonar_holder;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

        base::samples::Sonar sample;
        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.5, 0.5);

            /* denoising */
            cv::Mat cart_denoised;
            denoising::homomorphic_filter(cart_raw, cart_denoised, 2);
            // cv::Mat cart_denoised = rls.sliding_window(cart_raw);

            // /* cartesian roi image */
            // cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            // cv::resize(cart_drawable_area, cart_drawable_area, cart_denoised.size());
            // cv::Mat cart_mask = preprocessing::extract_roi_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);
            // cv::Mat cart_image;
            // cart_denoised.copyTo(cart_image, cart_mask);

            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::waitKey(30);
        }
        cv::waitKey(0);
    }

return 0;
}
