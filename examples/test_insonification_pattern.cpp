#include <iostream>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <base/samples/Sonar.hpp>
#include <base/Eigen.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Clustering.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/FrequencyDomain.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_pipeline/Application.hpp"

#include <opencv2/imgproc/imgproc.hpp>

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

cv::Mat insonification_correction (const cv::Mat& src, const cv::Mat& mask) {
    CV_Assert(mask.type() == CV_8U);

    cv::Mat dst = src.clone();

    // calculate the proportional mean of each image row
    std::vector<double> row_mean(src.rows, 0);
    for (size_t i = 0; i < src.rows; i++) {
        unsigned int num_points = cv::countNonZero(mask.row(i));
        if(num_points) {
            double value = cv::sum(src.row(i))[0] / num_points;
            row_mean[i] = std::isnan(value) ? 0 : value;
        }
    }

    // get the maximum mean between lines
    double max_mean = *std::max_element(row_mean.begin(), row_mean.end());

    // apply the insonification correction
    for (size_t i = 0; i < src.rows; i++) {
        if(row_mean[i]) {
            double factor = max_mean / row_mean[i];
            dst.row(i) *= factor;
        }
    }
    dst.setTo(1, dst > 1);
    return dst;
}

bool pipeline_detection(const cv::Mat& src, const cv::Mat& mask, std::pair<cv::Point2f, cv::Point2f>& best_model, double fit_rate, int min_valid_pixels, float angle, int step_angle, int step_scan) {
    cv::Point2f center(src.cols * 0.5, src.rows * 0.5);
    bool found = false;
    double best_score = 1;

    // find y-limits of mask
    // perform two iterators to find the upper and under mask limits on y-axis
    cv::Mat mask_y_center = mask.col(center.x).t() / 255;
    std::vector<float> data_mat(mask_y_center.datastart, mask_y_center.dataend);
    std::vector<float>::iterator min_idx = std::find(data_mat.begin(), data_mat.end(), 1);
    std::vector<float>::reverse_iterator max_idx = std::find(data_mat.rbegin(), data_mat.rend(), 1);
    if((min_idx == data_mat.end()) || (max_idx == data_mat.rend()))
        return found;
    int y_min = min_idx - data_mat.begin();
    int y_max = data_mat.rend() - max_idx;
    double f = (src.cols > src.rows) ? 0.9 : 0.4;
    min_valid_pixels = (y_max - y_min) * f;

    // pipeline searching using a sliding line segment
    int angle_range = tan(angle * M_PI / 180.0) * (y_max - y_min);
    for (int i = center.x - angle_range; i < center.x + angle_range; i+=step_angle) {
        for (int j = -mask.cols; j < mask.cols; j+=step_scan) {
            cv::Point2f p1(i + j, y_min);
            cv::Point2f p2(center.x + j, y_max);

            // check if the reference points p1 and p2 are contained within the sonar image
            bool contains_p1 = (p1.x >= 0) && (p1.x <= mask.cols);
            bool contains_p2 = (p2.x >= 0) && (p2.x <= mask.cols);
            if(!contains_p1 || !contains_p2)
                continue;

            // get all points on the line segment using a 8-connected pixels
            cv::LineIterator line_points(mask, p1, p2);

            // calculate the proportional average
            double accum_sum = 0;
            int valid_pixels = 0;
            for (int k = 0; k < line_points.count; k++, ++line_points) {
                accum_sum += src.at<float>(line_points.pos());
                if(mask.at<uchar>(line_points.pos()))
                    valid_pixels++;
            }

            // if there are valid pixels, evaluate the results
            if (!valid_pixels)
                continue;
            double avg = accum_sum / valid_pixels;
            if ((avg > fit_rate) || (valid_pixels < min_valid_pixels) || avg > best_score)
                continue;

            // store the best results
            best_model = std::make_pair(p1, p2);
            best_score = avg;
            found = true;
        }
    }
    return found;
}


int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/testsite.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    sonar_processing::SonarHolder sonar_holder;
    base::samples::Sonar sample;
    RLS rls(4);
    RLS rls2(4);
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");
        stream.set_current_sample_index(start_index);

        cv::Mat avgFrames;
        size_t counter = 0;
        bool calculated = false;

        while (stream.current_sample_index() < stream.total_samples()) {
            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.4, 0.4);

            if (avgFrames.empty()) {
                avgFrames = cv::Mat::zeros(cart_raw.size(), CV_32FC1);
            }

            if (!calculated) {
                cv::accumulate(cart_raw, avgFrames);
            }
            counter++;

            /* output results */
            cv::imshow("cart_raw", cart_raw);
            if((counter > 150) && (!calculated)) {
                avgFrames = avgFrames / counter;
                calculated = true;
            }
            if(calculated) {
                cv::imshow("avgFrames", avgFrames);
                cv::Mat res = cart_raw - avgFrames * 0.5;
                res.setTo(1, res > 1);
                res.setTo(0, res < 0);
                cv::imshow("res", res);

                /* drawable area */
                cv::Mat cart_mask = sonar_holder.cart_image_mask();
                cv::resize(cart_mask, cart_mask, cart_raw.size());

                /* denoising */
                cv::Mat cart_denoised = rls.sliding_window(res);

                /* cartesian roi image */
                cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_denoised, cart_mask, 0.1);

                /* insonification correction */
                cv::Mat cart_enhanced = insonification_correction(cart_denoised, cart_roi);

                /* filtering */
                cv::Mat cart_aux, cart_filtered;
                cart_enhanced.convertTo(cart_aux, CV_8U, 255);
                preprocessing::adaptive_clahe(cart_aux, cart_aux, 4);
                cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5,5));
                cart_filtered = cart_aux & cart_roi;
                cart_filtered.convertTo(cart_filtered, CV_32F, 1.0 / 255);
                cv::multiply(cart_filtered, cart_filtered, cart_filtered);

                cv::imshow("cart_denoised", cart_denoised);
                cv::imshow("cart_enhanced", cart_enhanced);
                cv::imshow("cart_filtered", cart_filtered);

                std::pair<cv::Point2f, cv::Point2f> best_model;
                bool found = pipeline_detection(cart_filtered, cart_roi, best_model, 0.15, 150, 30, 5, 5);
                if (found) {
                    cv::Point2f p1, p2;
                   p1 = best_model.first;
                   p2 = best_model.second;
                   cv::Mat cart_out;
                   cv::cvtColor(cart_raw, cart_out, CV_GRAY2BGR);
                   cart_out.convertTo(cart_out, CV_8UC3, 255);
                   cv::line(cart_out, p1, p2, cv::Scalar(0,255,0), 2, CV_AA);
                   cv::imshow("cart_out", cart_out);
                }

            }
            cv::waitKey(5);

            std::cout << " ==================== IDX: " << stream.current_sample_index() << std::endl;
        }

        cv::waitKey(0);
    }

return 0;
}