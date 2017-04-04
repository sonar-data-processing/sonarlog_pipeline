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
                // for (size_t k = 0; k < point_list.size(); k++) {
                    // if(inliers[k]) {
                        // dst.at<uchar>(point_list[k].y + i, point_list[k].x + j) = 255;
                    // }
                // }
                best_model = std::make_pair(cv::Point(best_model.first.x + j, best_model.first.y + i), cv::Point(best_model.second.x + j, best_model.second.y + i));
                cv::line(dst, best_model.first, best_model.second, cv::Scalar(255), 2);
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

std::vector<cv::Point2f> sliding_window2(cv::Mat src, size_t window_width, size_t window_height, size_t step, bool drawable = false) {
    cv::Mat current_window;

    std::vector<cv::Point2f> mc;

    // perform the sliding window
    for (size_t i = 0; i < src.rows; i += step) {
        if((i + window_height) > src.rows){ break; }
        for (size_t j = 0; j < src.cols; j += step) {
            if((j + window_width) > src.cols){ break; }
            // define the sliding window
            cv::Rect rect(j, i, window_width, window_height);
            cv::Mat subimage = src(rect);

            // std::cout<< " teste -1" << std::endl;

            // get all centroids
            std::vector< std::vector<cv::Point> > contours;
            cv::findContours(subimage.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            // cv::imshow(" out 1 ", subimage);
            // cv::waitKey();

            // std::cout<< " counter size " << contours.size() << std::endl;
            for (size_t k = 0; k < contours.size(); k++) {
                // std::cout<< "   counter " << k << " size " << contours[k].size() << std::endl;
                cv::Moments mu = cv::moments(cv::Mat(contours[k]));

                if(!std::isnan(mu.m00) && mu.m00 != 0)
                    mc.push_back(cv::Point2f( (mu.m10 / mu.m00) + j, (mu.m01 / mu.m00 ) + i ));

            }
        }
    }
    return mc;
}

cv::Mat frequency_domain_conversion (cv::Mat img) {
    int M = cv::getOptimalDFTSize( img.rows );
    int N = cv::getOptimalDFTSize( img.cols );
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);

    cv::dft(complexImg, complexImg);

    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);

    // crop the spectrum, if it has an odd number of rows or columns
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));

    int cx = mag.cols/2;
    int cy = mag.rows/2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    cv::Mat tmp;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(mag, mag, 0, 1, CV_MINMAX);
    return mag;
}

cv::Mat rotated_roi (cv::Mat source, float angle, cv::Rect roi) {
    cv::Point center(source.cols * 0.5, source.rows * 0.5);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1);
    cv::Mat rotated;                // cv::imshow("current_window", current_window);
                // cv::imshow("subimage", subimage);
                // cv::imshow("dst", dst);
                // cv::waitKey();
    cv::warpAffine(source, rotated, rot_mat, source.size());
    return rotated(roi);
}

cv::Mat insonification_correction (const cv::Mat& src, const cv::Mat& mask) {
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

    cv::imshow("dst1", dst);
    // output
    std::vector<float> data;
    data.assign((float*) dst.datastart, (float*) dst.dataend);
    std::replace_if(data.begin(), data.end(), std::bind2nd(std::greater<float>(), 1.0), 1.0);
    cv::Mat out(dst.rows, dst.cols, CV_32F, (void*) data.data());
    out.copyTo(dst);

    cv::imshow("dst2", dst);
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    dst.convertTo(dst, CV_8U, 255);
    cv::imshow("dst3", dst);
    double mean = cv::mean(dst, mask)[0];
    dst = dst < (mean * 0.5);
    return dst;
}

// convert between two different vector types
template <class SrcType, class DstType>
void convert3(std::vector<SrcType>& src, std::vector<DstType>& dst) {
    std::copy(src.begin(), src.end(), std::back_inserter(dst));
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.0.log",
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.2.log",
        DATA_PATH_STRING + "/logs/pipeline/pipeline_parallel.1.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(3);
    sonar_processing::SonarHolder sonar_holder;
    std::vector<cv::Mat> frames;

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

            cart_filtered.convertTo(cart_filtered, CV_32F, 1.0 / 255);
            // cart_denoised.convertTo(cart_denoised, CV_8U, 255);
            cv::Mat cart_thresh1;
            // cv::multiply(cart_filtered, cart_filtered, cart_filtered);
            cv::multiply(cart_filtered, cart_filtered, cart_thresh1, 1);
            cv::normalize(cart_thresh1, cart_thresh1, 0, 1, cv::NORM_MINMAX);
            cart_thresh1.convertTo(cart_thresh1, CV_8U, 255);
            cart_thresh1.copyTo(cart_filtered);

            /* local threshold */
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

            /* centroids */
            std::vector<cv::Point2f> centroids2 = sliding_window2(cart_thresh, 64, 64, 32, false);
            std::vector<cv::Point> centroids;
            convert3(centroids2, centroids);


            std::cout<< " centroid size " <<centroids.size() << std::endl;

            cv::Mat output_centroids;
            cv::cvtColor(cart_thresh, output_centroids, CV_GRAY2BGR);
            for (size_t k = 0; k < centroids.size(); k++) {
                // output_centroids.at<cv::Vec3b>(centroids[k]) = cv::Vec3b(0, 0, 255);
                cv::circle(output_centroids, centroids[k], 2, cv::Scalar(0,0,255), -1);
                // std::cout<< k << " point X Y: " << centroids[k] << std::endl;
            }



            // /* inverted image */
            // cv::Mat cart_inverted = (255 - cart_filtered) & cart_mask;
            // cv::imwrite("inverted.png", cart_inverted);
            //
            // /* segmentation */
            // cv::Mat cart_thresh = (cart_filtered > 128) & cart_mask;
            // cv::imwrite("threshold.png", cart_thresh);
            //
            // /* sliding window */
            // size_t window_width = 128, window_height = 128, step = 64;
            // cv::Mat cart_thresh2 = sliding_window(cart_thresh, window_width, window_height, step, false);
            //
            /* global ransac */
            // int num_pixels = cv::countNonZero(cart_thresh2);
            // if(num_pixels) {
                std::vector<cv::Point> point_list;
                point_list = centroids;
                // cv::findNonZero(cart_thresh2, point_list);

                // ransac fit line
                std::pair<cv::Point, cv::Point> best_model;
                std::vector<bool> inliers;
                double distance_average;
                fitLineRansac(point_list, 100, 20, 10, best_model, inliers, distance_average);
                if(!inliers.empty()) {
                    cv::Mat output;
                    cv::cvtColor(cart_thresh, output, cv::COLOR_GRAY2BGR);
                    drawStraightLine(output, best_model, cv::Scalar(0, 0, 255));
                    cv::imshow("output", output);
                }
            // }


            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_image", cart_image);
            cv::imshow("cart_filtered", cart_filtered);
            // cv::imshow("cart_inverted", cart_inverted);
            cv::imshow("cart_thresh", cart_thresh);
            cv::imshow("cart_thresh1", cart_thresh1);
            cv::imshow("plot_centroids", output_centroids);
            // cv::imshow("cart_thresh2", cart_thresh2);
            cv::waitKey(100);
        }
        cv::waitKey(0);
    }

return 0;
}
