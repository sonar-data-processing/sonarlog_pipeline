#include <iostream>
#include <cmath>
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

// cv::Mat perform_preprocessing(const cv::Mat& src, cv::Mat cart_roi_mask) {
//     /* gradient */
//     cv::Mat cart_image, dst;
//     // src.convertTo(cart_image, CV_8U, 255);
//     src.copyTo(cart_image);
//     cv::Mat cart_aux, cart_grad;
//     cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
//     preprocessing::gradient_filter(cart_image, cart_grad);
//     cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
//     cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
//     cart_grad -= cart_aux;
//     cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
//     cv::imshow("cart_grad", cart_grad);
//
//
//     /* mask */
//     cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
//     cv::Mat cart_grad2;
//     cart_grad.copyTo(cart_grad2, cart_roi_mask);
//     cv::medianBlur(cart_grad2, cart_grad2, 5);
//
//     /* threshold */
//     cv::Mat cart_thresh;
//     cv::threshold(cart_grad2, cart_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
//     cv::morphologyEx(cart_thresh, cart_thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)), cv::Point(-1, -1), 2);
//     preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(8, 8));
//     return cart_thresh;
// }

void preprocessing1(cv::Mat cart_image) {
    /* edge detector */
    cv::Mat edges;
    cart_image.convertTo(cart_image, CV_8U, 255);
    cv::Canny(cart_image, edges, 10, 60, 7);
    cv::imshow("canny", edges);

    /* morphological operations */
    cv::Mat close_edges;
    cv::morphologyEx(edges, close_edges, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5, 5)));
    cv::imshow("close_edges", close_edges);

    /* inverted image */
    cv::Mat inverted_edges = 255 - close_edges;
    cv::imshow("inverted_edges", inverted_edges);

    /* hough lines */
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(inverted_edges, lines, 1, CV_PI / 180, 200, 100, 80);
    cv::Mat colored;
    cv::cvtColor(cart_image, colored, cv::COLOR_GRAY2BGR);
    for(size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        if (cart_image.at<float>(cv::Point(l[0], l[1])) && cart_image.at<float>(cv::Point(l[2], l[3])))
            cv::line( colored, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, CV_AA);
    }
    cv::imshow("colored", colored);
}

cv::Mat getMean (std::vector<cv::Mat> frames) {
    if (frames.empty()) return cv::Mat();

    cv::Mat m = cv::Mat::zeros(frames[0].rows, frames[0].cols, CV_32FC1);
    cv::Mat temp;
    for (size_t i = 0; i < frames.size(); i++) {
        frames[i].copyTo(temp);
        m += temp;
    }

    return m / frames.size();
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
        std::vector<cv::Mat> accum;

        base::samples::Sonar sample;
        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.5, 0.5);

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);

            /* cartesian roi image */
            cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            cv::resize(cart_drawable_area, cart_drawable_area, cart_denoised.size());
            cv::Mat cart_mask = preprocessing::extract_roi_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);
            cv::Mat cart_image;
            cart_denoised.copyTo(cart_image, cart_mask);

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            cart_image.convertTo(cart_image, CV_8U, 255);
            preprocessing::adaptative_clahe(cart_image, cart_aux);
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5, 5));
            cv::morphologyEx(cart_mask, cart_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(9, 9)), cv::Point(-1, -1), 2);
            cart_aux.copyTo(cart_filtered, cart_mask);

            /* segmentation */
            cv::Mat cart_thresh, cart_aux2;
            cart_aux2 = cart_filtered < 50;
            cart_aux2.copyTo(cart_thresh, cart_mask);

            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_image", cart_image);
            cv::imshow("cart_filtered", cart_filtered);
            cv::imshow("cart_thresh", cart_thresh);
            // cv::imshow("cart_out", cart_out);
            cv::waitKey(30);
        }
        cv::waitKey(0);
    }

return 0;
}
