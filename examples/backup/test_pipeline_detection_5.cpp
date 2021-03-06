#include <iostream>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <base/samples/Sonar.hpp>
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

cv::Mat rotate_image (cv::Mat src, float angle, cv::Point center) {
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1);
    cv::Mat dst;
    cv::warpAffine(src, dst, rotation_matrix, src.size());
    return dst;
}

// bool pipeline_detection(cv::Mat src, cv::Mat mask, cv::Mat& matches) {
//     cv::Point center(src.cols * 0.5, src.rows * 0.5);
//     cv::Mat src_sum, mask_sum;
//     bool found = false;
//     matches = cv::Mat::zeros(src.size(), CV_8U);
//
//     for (float angle = -30; angle <= 30; angle += 1) {
//         // rotate the image and mask in defined angles
//         cv::Mat src_rotated = rotate_image(src, angle, center);
//         cv::Mat mask_rotated = rotate_image(mask, angle, center);
//
//         // reduce the matrices to get the accumulative sum
//         cv::reduce(src_rotated, src_sum, 0, CV_REDUCE_SUM, CV_32FC1);
//         cv::reduce(mask_rotated / 255, mask_sum, 0, CV_REDUCE_SUM, CV_32S);
//
//         // calculate the proportional average
//         for (size_t col_idx = 50; col_idx < src_sum.cols - 50; col_idx++) {
//             // check number of valid pixels in the mask
//             unsigned int valid_pixels = cv::sum(mask_sum.colRange(col_idx - 1, col_idx + 1))[0];
//             if (valid_pixels > 300) {
//                 double avg = cv::sum(src_sum.colRange(col_idx - 1, col_idx + 1))[0] / valid_pixels;
//                 // if the region average is too small, it may be the pipeline.
//                 // add it to match filter result
//                 if(avg < 0.15) {
//                     cv::Mat local_match = cv::Mat::zeros(src.size(), CV_8U);
//                     local_match.colRange(col_idx - 1, col_idx + 1) = 1;
//                     local_match = rotate_image(local_match, -angle, center);
//                     matches += local_match;
//                     found = true;
//                 }
//             }
//         }
//     }
//
//     // eliminate false alarm rates
//     matches = (matches > 3) & mask;
//
//     return found;
// }

// bool pipeline_detection(cv::Mat src, cv::Mat mask, cv::Mat& matches) {
//     cv::Point center(src.cols * 0.5, src.rows * 0.5);
//     cv::Mat src_sum, mask_sum;
//     bool found = false;
//     matches = cv::Mat::zeros(src.size(), CV_8U);
//
//     for (float angle = -30; angle <= 30; angle += 1) {
//         // rotate the image and mask in defined angles
//         cv::Mat src_rotated = rotate_image(src, angle, center);
//         cv::Mat mask_rotated = rotate_image(mask, angle, center);
//
//         // reduce the matrices to get the accumulative sum
//         cv::reduce(src_rotated, src_sum, 0, CV_REDUCE_SUM, CV_32FC1);
//         cv::reduce(mask_rotated / 255, mask_sum, 0, CV_REDUCE_SUM, CV_32S);
//
//         // calculate the proportional average
//         for (size_t col_idx = 50; col_idx < src_sum.cols - 50; col_idx++) {
//             // check number of valid pixels in the mask
//             unsigned int valid_pixels = mask_sum.at<int>(col_idx);
//             std::cout << "Valid pixels: " << valid_pixels << std::endl;
//             if (valid_pixels) {
//                 double avg = src_sum.at<float>(col_idx) / valid_pixels;
//                 std::cout << "Avg: " << avg << std::endl;
//
//                 cv::Mat out;
//                 cv::cvtColor(src_rotated, out, CV_GRAY2BGR);
//                 out.col(col_idx) = cv::Scalar(255,0,0);
//                 cv::imshow("out", out);
//                 cv::waitKey();
//
//                 // if the region average is too small, it may be the pipeline.
//                 // add it to match filter result
//                 // if(avg < 0.15) {
//                     // cv::Mat local_match = cv::Mat::zeros(src.size(), CV_8U);
//                     // local_match
//                     // local_match.colRange(col_idx - 1, col_idx + 1) = 1;
//                     // local_match = rotate_image(local_match, -angle, center);
//                     // matches += local_match;
//                     // found = true;
//                 // }
//             }
//         }
//     }
//
//     // eliminate false alarm rates
//     // matches = (matches > 3) & mask;
//
//     return found;
// }

// bool pipeline_detection(cv::Mat src, cv::Mat mask, cv::Mat& matches) {
//     cv::Point center(src.cols * 0.5, src.rows * 0.5);
//     cv::Mat src_sum, mask_sum;
//     bool found = false;
//     matches = cv::Mat::zeros(src.size(), CV_8U);
//
//     for (float angle = -30; angle <= 30; angle += 1) {
//         // rotate the image and mask in defined angles
//         cv::Mat src_rotated = rotate_image(src, angle, center);
//         cv::Mat mask_rotated = rotate_image(mask, angle, center);
//
//         // reduce the matrices to get the accumulative sum
//         cv::reduce(src_rotated, src_sum, 0, CV_REDUCE_SUM, CV_32FC1);
//         cv::reduce(mask_rotated / 255, mask_sum, 0, CV_REDUCE_SUM, CV_32S);
//
//         // calculate the proportional average
//         for (size_t col_idx = 50; col_idx < src_sum.cols - 50; col_idx++) {
//             // check number of valid pixels in the mask
//             unsigned int valid_pixels = mask_sum.at<int>(col_idx);
//             std::cout << "Valid pixels: " << valid_pixels << std::endl;
//             if (valid_pixels) {
//                 double avg = src_sum.at<float>(col_idx) / valid_pixels;
//                 std::cout << "Avg: " << avg << std::endl;
//
//                 cv::Mat out;
//                 cv::cvtColor(src_rotated, out, CV_GRAY2BGR);
//                 out.col(col_idx) = cv::Scalar(255,0,0);
//                 cv::imshow("out", out);
//                 cv::waitKey();
//
//                 // if the region average is too small, it may be the pipeline.
//                 // add it to match filter result
//                 // if(avg < 0.15) {
//                     // cv::Mat local_match = cv::Mat::zeros(src.size(), CV_8U);
//                     // local_match
//                     // local_match.colRange(col_idx - 1, col_idx + 1) = 1;
//                     // local_match = rotate_image(local_match, -angle, center);
//                     // matches += local_match;
//                     // found = true;
//                 // }
//             }
//         }
//     }
//
//     // eliminate false alarm rates
//     // matches = (matches > 3) & mask;
//
//     return found;
// }


void line_equation(cv::Point p1, cv::Point p2, double& m, double& b) {
    if (p1.x != p2.x) {
        m = (double) (p1.y - p2.y) / (double) (p1.x - p2.x);
        b = p1.y - (m * p1.x);
    } else {
        m = 0;
        b = p2.y;
    }
}

bool pipeline_detection(const cv::Mat& src, const cv::Mat& mask, cv::Mat& matches, float angle, int step_angle, int step_scan) {
    cv::Point center(src.cols * 0.5, src.rows * 0.5);
    matches = cv::Mat::zeros(src.size(), CV_8U);
    bool found = false;
    int best_match = 0;
    double best_score = 1;
    cv::Point b1, b2;

    // find y-limits of mask
    cv::Mat mask_y_center = mask.col(center.x);
    int y_min = -1, y_max = -1;
    bool found_y_min = false, found_y_max = false;
    int i = 0, j = mask.rows;
    while(!found_y_min || !found_y_max) {
        if (!found_y_min && mask_y_center.at<uchar>(i++) == 255) {
            y_min = i;
            found_y_min = true;
        }
        if (!found_y_max && mask_y_center.at<uchar>(j--) == 255) {
            y_max = j;
            found_y_max = true;
        }
    }

    // pipeline searching
    int angle_range = tan(angle * M_PI / 180.0) * (y_max - y_min);

    for (size_t i = center.x - angle_range; i < center.x + angle_range; i+=step_angle) {
        for (int j = -mask.cols; j < mask.cols; j+=step_scan) {
            cv::Point p1(i + j, y_min);
            cv::Point p2(center.x + j, y_max);

            bool contains_p1 = !(p1.x < 0 || p1.x > mask.cols);
            bool contains_p2 = !(p2.x < 0 || p2.x > mask.cols);

            if (contains_p1 && contains_p2) {
                cv::LineIterator line_points(mask, p1, p2);

                double accum_sum = 0;
                int valid_pixels = 0;
                for (int k = 0; k < line_points.count; k++, ++line_points) {
                    accum_sum += src.at<float>(line_points.pos());
                    if(mask.at<uchar>(line_points.pos())== 255)
                        valid_pixels++;
                }

                if(valid_pixels) {
                    double avg = accum_sum / valid_pixels;
                    if (avg < 0.15 && valid_pixels > 150 && best_score > avg) {
                        // best_match = valid_pixels;
                        b1 = p1;
                        b2 = p2;
                        best_score = avg;
                        // std::cout << "Avg: " << avg << std::endl;
                        // std::cout << "Valid Pixels: " << valid_pixels << std::endl;
                        // // matches = cv::Mat::zeros(src.size(), CV_8U);
                        // cv::Mat out;
                        // cv::cvtColor(src, out, CV_GRAY2BGR);
                        // cv::line(out, p1, p2, cv::Scalar(255,0,0), 2);
                        // cv::imshow("Matches", out);
                        // cv::waitKey(100);

                    }
                }



            }
        }
    }

    if(best_score) {
        cv::Mat out;
        cv::cvtColor(src, out, CV_GRAY2BGR);
        cv::line(out, b1, b2, cv::Scalar(255,0,0), 2);
        cv::imshow("out", out);
        cv::waitKey(5);
    }
    // // matches = cv::Mat::zeros(src.size(), CV_8U);
    // std::cout << "Avg: " << avg << std::endl;
    // std::cout << "Valid Pixels: " << valid_pixels << std::endl;
    // cv::imshow("Matches", out);
    // cv::waitKey(100);

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


int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.2.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.1.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.0.log",
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(3);
    sonar_processing::SonarHolder sonar_holder;
    base::samples::Sonar sample;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

        int count = 0;
        while (stream.current_sample_index() < stream.total_samples()) {

            clock_t tStart = clock();

            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.4, 0.4);

            /* drawable area */
            cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            cv::resize(cart_drawable_area, cart_drawable_area, cart_raw.size());

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);

            /* cartesian roi image */
            cv::Mat cart_mask = preprocessing::extract_cartesian_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);

            /* insonification correction */
            cv::Mat cart_enhanced = insonification_correction(cart_denoised, cart_mask);

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            cart_enhanced.convertTo(cart_aux, CV_8U, 255);
            preprocessing::adaptive_clahe(cart_aux, cart_aux, 4);
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5,5));
            cart_filtered = cart_aux & cart_mask;
            cart_filtered.convertTo(cart_filtered, CV_32F, 1.0 / 255);
            cv::multiply(cart_filtered, cart_filtered, cart_filtered);

            /* line processing */
            cv::Mat matches;
            if (++count > rls.getWindow_size()) {
            //     // std::cout << "Frame: " << count << std::endl;
            //     // cv::imshow("cart_raw", cart_raw);
            //     // cv::imshow("cart_denoised", cart_denoised);
            //     // cv::imshow("cart_enhanced", cart_enhanced);
            //     // cv::imshow("cart_filtered", cart_filtered);
            //
                // pipeline detection
                if (pipeline_detection(cart_filtered, cart_mask, matches, 30, 10, 5)) {
                // if (pipeline_detection(cart_filtered, cart_mask, matches) {
                //     cv::imshow("matches", matches);
                }
            //     // cv::waitKey(5);
            }
            clock_t tEnd = clock();
            double elapsed_secs = double (tEnd - tStart) / CLOCKS_PER_SEC;
            std::cout << " ==================== FPS: " << (1.0 / elapsed_secs) << std::endl;
        }

        cv::waitKey(0);
    }

return 0;
}
