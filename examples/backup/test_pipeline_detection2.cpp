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

/**
  * Uses the ransac algorithm to estimate lines in 2d point set.
  *
  * @param points: the 2d point set
  * @param iterations: number of iterations
  * @param min_inliers: minimum number of points to fit
  * @param threshold: max inlier distance to the model
  * @param best_model - best model that could be found
  * @param inliers: mask containing the inliers
  * @param distance_average: distance average from inliers to best model
  * @return best_error, outlier in percent. if returns 1.0, no valid model could be found
*/
void fitLineRansac (    const std::vector<cv::Point>& points,
                        unsigned int iterations,
                        unsigned int min_inliers,
                        double distance_threshold,
                        std::pair<cv::Point, cv::Point>& best_model,
                        std::vector<bool>& inliers,
                        double &distance_average) {

    int best_num_inliers = 0;
    double best_distance_average = 0;

    int num_points = points.size();
    if ((num_points < 2) || (num_points < min_inliers))
        return;

    cv::RNG rng;
    for (size_t n = 0; n < iterations; n++) {
        // pick two different random points
        cv::Point p1, p2;
        do {
            int idx1 = rng(num_points);
            int idx2 = rng(num_points);
            p1 = points[idx1];
            p2 = points[idx2];
        } while (p1 == p2);

        // line equation: (y1 - y2)X + (x2 - x1)Y + x1y2 - x2y1 = 0
        float a = static_cast<float>(p1.y - p2.y);
        float b = static_cast<float>(p2.x - p1.x);
        float c = static_cast<float>(p1.x * p2.y - p2.x * p1.y);

        // line fit evaluation
        float scale = 1 / std::sqrt(a * a + b * b);
        double distance_average = 0;
        std::vector<bool> current_inliers(num_points, false);

        for (size_t i = 0; i < num_points; i++) {
            // distance from the point to the line
            cv::Point p0 = points[i];
            double distance_to_line = std::abs(a * p0.x + b * p0.y + c) * scale;

            // check for inlier points
            if (distance_to_line < distance_threshold) {
                distance_average += distance_to_line;
                current_inliers[i] = true;
            }
        }

        // check fit rate
        unsigned int num_inliers = std::count(current_inliers.begin(), current_inliers.end(), true);
        if ((num_inliers > min_inliers) && (num_inliers > best_num_inliers)) {
            best_num_inliers = num_inliers;
            best_distance_average = distance_average;
            best_model = std::make_pair(p1, p2);
            inliers = current_inliers;
        }
    }
}

void drawStraightLine(cv::Mat& img, const std::pair<cv::Point, cv::Point>& best_model, cv::Scalar color) {
    cv::Point p1 = best_model.first;
    cv::Point p2 = best_model.second;
    cv::Point p(0,0), q(img.cols, img.rows);

    if (p1.x != p2.x) {
        double m = (double) (p1.y - p2.y) / (double) (p1.x - p2.x);
        double b = p1.y - (m * p1.x);
        p.y = m * p.x + b;
        q.y = m * q.x + b;
    } else {
        p.x = q.x = p2.x;
        p.y = 0;
        q.y = img.rows;
    }
    cv::clipLine(img.size(), p, q);

    // output drawing
    cv::line(img, p, q, color, 2, CV_AA);
    cv::circle(img, p1, 4, cv::Scalar(255,0,0), -1);
    cv::circle(img, p2, 4, cv::Scalar(255,0,0), -1);
}

void process_sliding_window(cv::Mat sliding_window, std::pair<cv::Point, cv::Point> &best_model, std::vector<bool>& inliers) {
    // count white pixels
    unsigned int count_zeros = cv::countNonZero(sliding_window);

    if (count_zeros) {
        std::vector<cv::Point> point_list;
        cv::findNonZero(sliding_window, point_list);

        // run ransac fit line
        unsigned int iterations = 50, min_inliers = 500;
        double distance_threshold = 10, distance_average;
        fitLineRansac(point_list, iterations, min_inliers, distance_threshold, best_model, inliers, distance_average);

        // if find best model, print it
        if (!inliers.empty()) {
            cv::Mat cart_ransac;
            cv::cvtColor(sliding_window, cart_ransac, cv::COLOR_GRAY2BGR);

            for (size_t i = 0; i < inliers.size(); i++)
                if(inliers[i])
                    cart_ransac.at<cv::Vec3b>(point_list[i]) = cv::Vec3b(0, 255, 0);

            drawStraightLine(cart_ransac, best_model, cv::Scalar(0,0,255));
            cv::imshow("ransac", cart_ransac);
        }
    }
}

cv::Mat sliding_window(cv::Mat src, size_t window_width, size_t window_height, size_t step, bool drawable = false) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat current_window;
    int counter = 0;

    // perform the sliding window
    for (size_t i = 0; i < src.rows; i += step) {
        if((i + window_height) > src.rows){ break; }
        for (size_t j = 0; j < src.cols; j += step) {
            if((j + window_width) > src.cols){ break; }
            // define the sliding window
            cv::Rect rect(j, i, window_width, window_height);
            cv::Mat subimage = src(rect);

            // process the sliding window using a local ransac
            std::vector<bool> inliers;
            std::pair<cv::Point, cv::Point> best_model;
            process_sliding_window(subimage, best_model, inliers);

            // if the local ransac parameters are satisfied, pass the inliers to new segmented image
            if (!inliers.empty()) {
                counter++;
                std::vector<cv::Point> point_list;
                cv::findNonZero(subimage, point_list);
                for (size_t k = 0; k < point_list.size(); k++) {
                    if(inliers[k]) {
                        dst.at<uchar>(point_list[k].y + i, point_list[k].x + j) = 255;
                    }
                }
                // best_model = std::make_pair(cv::Point(best_model.first.x + j, best_model.first.y + i), cv::Point(best_model.second.x + j, best_model.second.y + i));
                // cv::line(dst, best_model.first, best_model.second, cv::Scalar(255), 2);
            }
            // output
            if(drawable) {
                cv::cvtColor(src, current_window, cv::COLOR_GRAY2BGR);
                cv::rectangle(current_window, rect, cv::Scalar(0,0,255));
                cv::imshow("current_window", current_window);
                cv::imshow("subimage", subimage);
                cv::imshow("dst", dst);
                cv::waitKey();
            }
        }
    }


    std::cout << "Counter: " << counter << std::endl;
    return dst;
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.0.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.1.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.2.log",
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
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

            /* drawable area */
            cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            cv::resize(cart_drawable_area, cart_drawable_area, cart_raw.size());

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);

            /* cartesian roi image */
            cv::Mat cart_mask = preprocessing::extract_roi_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);
            cv::Mat cart_image;
            cart_denoised.copyTo(cart_image, cart_mask);

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            cart_image.convertTo(cart_aux, CV_8U, 255);
            preprocessing::adaptive_clahe(cart_aux, cart_aux, 10);
            cart_aux = cart_aux & cart_mask;
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5, 5));
            cv::morphologyEx(cart_mask, cart_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(9, 9)), cv::Point(-1, -1), 2);
            cart_aux.copyTo(cart_filtered, cart_mask);

            // /* segmentation */
            // cv::Mat cart_thresh = (cart_filtered > 128) & cart_mask;
            //
            // /* sliding window */
            // size_t window_width = 128, window_height = 128, step = 64;
            // cv::Mat cart_thresh2 = sliding_window(cart_thresh, window_width, window_height, step, false);

            /* shadow enhancement */
            cv::Mat cart_enhanced;
            cart_filtered.convertTo(cart_enhanced, CV_32FC1, 1.0 / 255);
            cv::multiply(cart_enhanced, cart_image, cart_enhanced);

            /* segmentation */
            cv::Mat cart_thresh = cv::Mat::zeros(cart_filtered.size(), cart_filtered.type());
            for (size_t j = 0; j < cart_filtered.rows; j++) {
                unsigned int num_points = cv::countNonZero(cart_mask.row(j));
                if(num_points) {
                    double value = cv::sum(cart_filtered.row(j))[0] / num_points;
                    double row_mean = std::isnan(value) ? 0 : value;

                    cart_thresh.row(j) = cart_filtered.row(j) < (row_mean * 0.5);
                }
            }
            cart_thresh = cart_thresh & cart_mask;
            preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(10, 10), CV_RETR_LIST);

            /* blob removal */
            int threshold = 40;
            std::vector< std::vector<cv::Point> > contours, new_contours;
            cv::findContours(cart_thresh.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            for (size_t j = 0; j < contours.size(); j++) {
                int min=99999, max = -99999;
                for (size_t k = 0; k < contours[j].size(); k++) {
                    if(contours[j][k].x < min)
                        min = contours[j][k].x;

                    if(contours[j][k].x > max)
                        max = contours[j][k].x;
                }
                if((max - min) < threshold)
                    new_contours.push_back(contours[j]);
            }

            cv::Mat new_image = cv::Mat::zeros(cart_image.size(), CV_8UC1);
            cv::drawContours(new_image, new_contours, -1, cv::Scalar(255), CV_FILLED);

            /* sliding window */
            size_t window_width = 128, window_height = 128, step = 64;
            cv::Mat cart_thresh2 = sliding_window(cart_thresh, window_width, window_height, step, false);


            // cv::Mat output;
            // cv::cvtColor(new_image, output, CV_GRAY2BGR);
            //
            //
            // unsigned int count_zeros = cv::countNonZero(new_image);
            //
            //
            // if (count_zeros) {
            //     std::vector<cv::Point> point_list;
            //     cv::findNonZero(new_image, point_list);
            //
            //     // run ransac fit line
            //     unsigned int iterations = 200, min_inliers = 250;
            //     double distance_threshold = 2, distance_average;
            //     std::pair<cv::Point, cv::Point> best_model;
            //     std::vector<bool> inliers;
            //     fitLineRansac(point_list, iterations, min_inliers, distance_threshold, best_model, inliers, distance_average);
            //     if(!inliers.empty()) {
            //         drawStraightLine(output, best_model, cv::Scalar(255,0,0));
            //         for (size_t z = 0; z < inliers.size(); z++)
            //             if(inliers[z])
            //                 output.at<cv::Vec3b>(point_list[z]) = cv::Vec3b(0, 255, 0);
            //     }
            //
            // }


            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_image", cart_image);
            cv::imshow("cart_filtered", cart_filtered);
            cv::imshow("cart_enhanced", cart_enhanced);
            cv::imshow("cart_thresh", cart_thresh);
            cv::imshow("cart_thresh2", cart_thresh2);
            cv::imshow("new_image", new_image);
            // cv::imshow("output", output);
            cv::waitKey();
        }
        cv::waitKey(0);
    }

return 0;
}
