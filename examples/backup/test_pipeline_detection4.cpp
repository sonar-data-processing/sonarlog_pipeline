#include <iostream>
#include <cmath>
#include <algorithm>
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

bool line_processing(cv::Mat src, cv::Mat mask, cv::Mat& mask_out) {
    // get the mask centroid
    cv::Moments m = cv::moments(mask, true);
    cv::Point centroid(m.m10 / m.m00, m.m01/m.m00);

    // resulting mask
    mask_out = cv::Mat::zeros(mask.size(), mask.type());
    bool found = false;

    // run the algorithm
    for (size_t i = 0; i < src.cols; i += 10) {
        cv::Point p(i, centroid.y);

        // point inside the mask
        if(mask.at<uchar>(p)) {

            for (float angle = -30; angle <= 30; angle += 10) {
                cv::Mat src_rotated = rotate_image(src, angle, p);
                cv::Mat mask_rotated = rotate_image(mask, angle, p);

                // calculate the proportional mean
                unsigned int num_pixels = cv::countNonZero(mask_rotated.colRange(i-2, i+2));
                double mean = cv::sum(src_rotated.colRange(i-2, i+2))[0] / num_pixels;

                // output view
                if(mean < 0.2 && num_pixels > 100) {
                    cv::Mat out_rotated = rotate_image(mask_out, angle, p);
                    out_rotated.colRange(i-2, i+2) = 255;
                    out_rotated = rotate_image(out_rotated, -angle, p);
                    mask_out = out_rotated & mask;
                    found = true;

                    // output info
                    std::cout << "Mean: " << mean << std::endl;
                    std::cout << "Num pixels: " << num_pixels << std::endl;
                    src_rotated.col(i) = 255;
                    cv::cvtColor(src_rotated, src_rotated, CV_GRAY2BGR);
                    cv::circle(src_rotated, p, 2, cv::Scalar(0,0,255), -1);
                    cv::imshow("rotated", src_rotated);
                    cv::imshow("mask_out", mask_out);
                }
            }
        }
    }
    return found;
}

bool line_processing2(cv::Mat src, cv::Mat mask) {
    cv::Point center(src.cols * 0.5, src.rows * 0.5);
    cv::Mat src_sum, mask_sum;
    bool found = false;

    for (float angle = -30; angle < 30; angle += 3) {
        // rotate the image and mask in defined angles
        cv::Mat src_rotated = rotate_image(src, angle, center);
        cv::Mat mask_rotated = rotate_image(mask, angle, center);

        // reduce the matrices to get the accumulative sum
        cv::reduce(src_rotated, src_sum, 0, CV_REDUCE_SUM, CV_32FC1);
        cv::reduce(mask_rotated / 255, mask_sum, 0, CV_REDUCE_SUM, CV_32S);

        // calculate the proportional means
        for (size_t i = 30; i < mask_sum.cols - 30; i += 10) {
            double num_pixels = cv::sum(mask_sum.colRange(i-2, i+2))[0];
            if(num_pixels > 500){
                double mean = cv::sum(src_sum.colRange(i-2,i+2))[0] / num_pixels;
                if (mean < 0.15) {
                    cv::Mat out;
                    cv::cvtColor(src_rotated, out, CV_GRAY2BGR);
                    out.col(i) = 255;
                    // show_out = rotate_image(out, -angle, center);

                    cv::imshow("out",out);
                    std::cout << "Num pixels: " << num_pixels << ", Mean: " << mean << std::endl;
                    // cv::waitKey();
                    // found = true;
                    // output info
                }
            }
        }

        // cv::imshow("src_rotated", src_rotated);
        // cv::imshow("mask_rotated", mask_rotated);
        // cv::waitKey();
    }
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
    dst.setTo(0, dst < 0);
    return dst;
}


int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.1.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.0.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.2.log",
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
            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.5, 0.5);

            /* drawable area */
            cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            cv::resize(cart_drawable_area, cart_drawable_area, cart_raw.size());

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);

            /* cartesian roi image */
            cv::Mat cart_mask = preprocessing::extract_cartesian_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);
            cv::Mat cart_image;
            cart_denoised.copyTo(cart_image, cart_mask);

            /* insonification correction */
            cv::Mat cart_enhanced = insonification_correction(cart_image, cart_mask);

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            cart_enhanced.convertTo(cart_aux, CV_8U, 255);
            preprocessing::adaptive_clahe(cart_aux, cart_aux, 10);
            cart_aux = cart_aux & cart_mask;
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5, 5));
            cv::morphologyEx(cart_mask, cart_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(9, 9)), cv::Point(-1, -1), 2);
            cart_mask = (cart_mask == 255);
            cart_filtered = cart_aux & cart_mask;
            cart_filtered.convertTo(cart_filtered, CV_32F, 1.0 / 255);
            cv::multiply(cart_filtered, cart_filtered, cart_filtered);

            /* line processing */
            if (count++ >= rls.getWindow_size()) {
                std::cout << "Frame: " << count << std::endl;
                cv::imshow("cart_raw", cart_raw);
                cv::imshow("cart_denoised", cart_denoised);
                cv::imshow("cart_mask", cart_mask);
                cv::imshow("cart_image", cart_image);
                cv::imshow("cart_enhanced", cart_enhanced);
                cv::imshow("cart_filtered", cart_filtered);
                bool found = line_processing2(cart_filtered, cart_mask);
                cv::waitKey(20);
            }
        }
        cv::waitKey(0);
    }

return 0;
}
