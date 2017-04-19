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

bool pipeline_detection(const cv::Mat& src, const cv::Mat& mask, std::pair<cv::Point2f, cv::Point2f>& best_model, double fit_rate, float angle, int step_angle, int step_scan) {
    bool found = false;
    double best_score = 1;

    // find mask limits
    cv::Mat points;
    cv::findNonZero(mask, points);
    cv::Rect rect = cv::boundingRect(points);

    double f = (rect.width > rect.height) ? 0.9 : 0.5;
    size_t min_valid_pixels = (rect.height) * f;
    // cv::Mat matches = cv::Mat::zeros(src.size(), CV_32F);
    //
    // double mean = 0;
    // int counter = 0;

    // pipeline searching using a sliding line segment
    cv::Point2f center(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    int angle_range = tan(angle * M_PI / 180.0) * rect.height;
    int corner_x = rect.x + rect.width;
    for (int i = (center.x - angle_range); i < (center.x + angle_range); i += step_angle) {
        for (int j = -mask.cols; j < mask.cols; j += step_scan) {
            cv::Point p1(i + j, rect.y);
            cv::Point p2(center.x + j, rect.y + rect.height - 1);

            // check if the reference points p1 and p2 are contained within the mask image
            if(!rect.contains(p1) && !rect.contains(p2)) {
                if(p1.x > corner_x && p2.x > corner_x)
                    break;
                else
                    continue;
            }

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

            // mean += (accum_sum / valid_pixels);
            // counter++;

            // if there are valid pixels, evaluate the results
            if (valid_pixels < min_valid_pixels)
                continue;
            double avg = accum_sum / valid_pixels;

            // std::cout << "Valid pixels: " << valid_pixels << std::endl;
            // std::cout << "Min Valid pixels: " << min_valid_pixels << std::endl;
            //
            // cv::Mat draw;
            // cv::cvtColor(src, draw, CV_GRAY2BGR);
            // cv::line(draw, p1, p2, cv::Scalar(0,255,0), 1, CV_AA);
            // cv::imshow("draw", draw);
            // std::cout << "Avg: " << avg << std::endl;
            // cv::waitKey(1);


            if ((avg > fit_rate) || (valid_pixels < min_valid_pixels) || avg > best_score)
                continue;

            // store the best results
            // cv::clipLine(rect, p1, p2);
            best_model = std::make_pair(cv::Point2f(p1.x, p1.y), cv::Point2f(p2.x, p2.y));
            best_score = avg;
            found = true;

            // cv::LineIterator line_points2(mask, p1, p2);
            // for (int k = 0; k < line_points2.count; k++, ++line_points2) {
                // matches.at<float>(line_points2.pos()) += 1;
            // }
        }
    }
    std::cout << "Best score: " << best_score << std::endl;
    // std::cout << "Mean: " << mean / counter << std::endl;
    // std::cout << "Best score: " << best_score << std::endl;
    // cv::normalize(matches, matches, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("matches", matches);
    // cv::waitKey();

    return found;
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

base::Vector2d getWorldPoint(const cv::Point2f& p, cv::Size size, float range) {
    // convert from image to cartesian coordinates
    cv::Point2f origin(size.width / 2, size.height - 1);
    cv::Point2f q(origin.y - p.y, origin.x - p.x);

    // sonar resolution
    float sonar_resolution = range / size.height;

    // 3d world coordinates
    float x = q.x * sonar_resolution;
    base::Angle angle = base::Angle::fromRad(atan2(q.y, q.x));
    float y = tan(angle.rad) * x;

    // output
    return base::Vector2d(x, y);
}

// bool checkFrameQuality(const cv::Mat& frame, const cv::Mat& mask) {
//     // find mask limits
//     cv::Mat points;
//     cv::findNonZero(mask, points);
//     cv::Rect rect = cv::boundingRect(points);
//
//     // superior
//     cv::Mat superior = frame(cv::Rect(rect.x, rect.y, rect.width, rect.height / 2));
//     cv::Mat inferior = frame(cv::Rect(rect.x, rect.y + rect.height / 2, rect.width, rect.height / 2));
//
//     double avg = cv::sum(superior)[0] / superior.total();
//
//
//
//     // int valid_pixels = cv::countNonZero(mask);
//     // double avg = cv::sum(frame)[0] / valid_pixels;
//     std::cout << "AVG: " << avg << std::endl;
//
//     cv::imshow("frame", frame);
//     cv::imshow("mask", mask);
//     cv::Mat test = cv::Mat::zeros(mask.size(), CV_8U);
//     cv::rectangle(test, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255), 2);
//     cv::imshow("test", test);
//     cv::imshow("superior", superior);
//     cv::imshow("inferior", inferior);
//     // cv::waitKey();
//
//
//
//     return true;
//
// }

bool checkFrameQuality(const cv::Mat& frame, const cv::Mat& mask) {
    // find mask limits
    cv::Mat points;
    cv::findNonZero(mask, points);
    cv::Rect rect = cv::boundingRect(points);

    // superior
    cv::Mat superior = frame(cv::Rect(rect.x, rect.y, rect.width, rect.height / 2));
    double avg = cv::sum(superior)[0] / superior.total();
    std::cout << "Avg: " << avg << std::endl;
    return (avg > 0.16);
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/testsite.0.log",
        // DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.0.log",
        // DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.1.log",
        // DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.2.log",
        // DATA_PATH_STRING + "/logs/pipeline-front.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(4);
    sonar_processing::SonarHolder sonar_holder;
    base::samples::Sonar sample;
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");
        stream.set_current_sample_index(start_index);
        bool scan_vertical = true;

        while (stream.current_sample_index() < stream.total_samples()) {

            clock_t tStart = clock();

            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.4, 0.4);

            /* drawable area */
            cv::Mat cart_mask = sonar_holder.cart_image_mask();
            cv::resize(cart_mask, cart_mask, cart_raw.size());

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);
            if(rls.getBuffer_size() < rls.getWindow_size()) continue;

            /* cartesian roi image */
            cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_denoised, cart_mask, 0.2);

            cv::imshow("cart_denoised", cart_denoised);

            /* filtering */
            cv::Mat cart_test;
            cart_denoised.convertTo(cart_test, CV_8U, 255);
            cart_test = cart_test & cart_roi;
            cart_test.convertTo(cart_test, CV_32F, 1.0 / 255);
            cv::imshow("cart_test", cart_test);

            if(!checkFrameQuality(cart_test, cart_roi)) continue;

            /* insonification correction */
            cv::Mat cart_enhanced = insonification_correction(cart_denoised, cart_roi);



            //
            // cv::Mat cart_test2;
            // cv::blur(cart_test, cart_test2, cv::Size(1, 20));
            // //
            // cart_test.convertTo(cart_test, CV_32F, 1.0 / 255);
            // cart_test2.convertTo(cart_test2, CV_32F, 1.0 / 255);
            // // cv::imshow("cart_test2", cart_test2);
            // //
            // cv::Mat cart_test3;
            // cv::multiply(cart_test, cart_test2, cart_test3);
            // cart_test3.convertTo(cart_test3, CV_8U, 255);
            // preprocessing::adaptive_clahe(cart_test3, cart_test3, 3);
            // cv::imshow("cart_test3", cart_test3);
            // cart_test3.convertTo(cart_test3, CV_32F, 1.0 / 255);



            // cv::Mat cart_aux, cart_filtered;
            // cart_enhanced.convertTo(cart_aux, CV_8U, 255);
            // // cv::blur(cart_aux, cart_aux, cv::Size(5,5));
            // cv::GaussianBlur(cart_aux, cart_aux, cv::Size(5,5), 0, 0);
            // preprocessing::adaptive_clahe(cart_aux, cart_aux, 4);
            // cart_filtered = cart_aux & cart_roi;
            // cart_filtered.convertTo(cart_filtered, CV_32F, 1.0 / 255);
            // cv::multiply(cart_filtered, cart_filtered, cart_filtered);

            // /* output results */
            // cv::imshow("cart_raw", cart_raw);
            // cv::imshow("cart_roi", cart_roi);
            // cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_enhanced", cart_enhanced);
            // cv::imshow("cart_test", cart_test);
            // cv::imshow("cart_test2", cart_test2);
            // cv::imshow("cart_test3", cart_test3);

            // // pipeline detection
            // std::pair<cv::Point2f, cv::Point2f> best_model;
            // bool found;
            // if(scan_vertical){
            //     found = pipeline_detection(cart_filtered, cart_roi, best_model, 0.15, 45, 5, 5);
            //     scan_vertical = found;
            // } else {
            //     found = pipeline_detection(cart_filtered.t(), cart_roi.t(), best_model, 0.25, 45, 5, 5);
            //     scan_vertical = !found;
            // }
            // std::cout << "Vertical: " << scan_vertical << ", Horizontal: " << !scan_vertical << std::endl;
            //
            // if (found) {
            //     cv::Point2f p1, p2;
            //     if (scan_vertical) {
            //         p1 = best_model.first;
            //         p2 = best_model.second;
            //     } else {
            //         p1 = cv::Point2f(best_model.first.y, best_model.first.x);
            //         p2 = cv::Point2f(best_model.second.y, best_model.second.x);
            //     }
            //
            //     cv::Mat cart_out;
            //     cv::cvtColor(cart_raw, cart_out, CV_GRAY2BGR);
            //     cart_out.convertTo(cart_out, CV_8UC3, 255);
            //     cv::line(cart_out, p1, p2, cv::Scalar(0,255,0), 2, CV_AA);
            //     cv::line(cart_out, cv::Point(0,cart_out.rows / 2), cv::Point(cart_out.cols -1,cart_out.rows / 2), cv::Scalar(0,0,255), 1, CV_AA);
            //     cv::line(cart_out, cv::Point(cart_out.cols / 2, 0), cv::Point(cart_out.cols / 2, cart_out.rows - 1), cv::Scalar(0,0,255), 1, CV_AA);
            //     cv::imshow("cart_out", cart_out);
            // }

            cv::waitKey(30);
            clock_t tEnd = clock();
            double elapsed_secs = double (tEnd - tStart) / CLOCKS_PER_SEC;
            std::cout << " ==================== FPS: " << (1.0 / elapsed_secs) << std::endl;
            std::cout << " ==================== IDX: " << stream.current_sample_index() << std::endl;
        }

        cv::waitKey(0);
    }

return 0;
}
