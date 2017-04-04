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
        int num_inliers = 0;
        double distance_average = 0;
        std::vector<bool> current_inliers(num_points, false);

        for (size_t i = 0; i < num_points; i++) {
            // distance from the point to the line
            cv::Point p0 = points[i];
            double distance_to_line = std::abs(a * p0.x + b * p0.y + c) * scale;

            // check for inlier points
            if (distance_to_line < distance_threshold) {
                distance_average += distance_to_line;
                num_inliers++;
                current_inliers[i] = true;
            }
        }

        // check fit rate
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
    double slope = (p2.y - p1.y) / (double) (p2.x - p1.x + std::numeric_limits<double>::epsilon());
    cv::Point p(0,0), q(img.cols, img.rows);
    p.y = -(p1.x - p.x) * slope + p1.y;
    q.y = -(p2.x - q.x) * slope + p2.y;
    cv::line(img, p, q, color, 2);
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
        DATA_PATH_STRING + "/logs/pipeline-front.1.log",
        DATA_PATH_STRING + "/logs/pipeline-parallel.0.log",
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
            preprocessing::adaptive_clahe(cart_image, cart_aux);
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5, 5));
            cv::morphologyEx(cart_mask, cart_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(9, 9)), cv::Point(-1, -1), 2);
            cart_aux.copyTo(cart_filtered, cart_mask);

            /* segmentation */
            cv::Mat cart_thresh, cart_aux2;
            cart_aux2 = cart_filtered < 50;
            cart_aux2.copyTo(cart_thresh, cart_mask);

            /* 2d point list */
            std::vector<cv::Point> point_list;
            cv::findNonZero(cart_thresh, point_list);

            /* ransac fit line */
            std::pair<cv::Point, cv::Point> best_model;
            std::vector<bool> inliers;
            double distance_average;
            fitLineRansac(point_list, 100, 100, 10, best_model, inliers, distance_average);

            cv::Mat cart_out, cart_out2;
            cv::cvtColor(cart_thresh, cart_out, cv::COLOR_GRAY2BGR);
            cv::cvtColor(cart_raw, cart_out2, cv::COLOR_GRAY2BGR);
            for (size_t i = 0; i < inliers.size(); i++)
                if(inliers[i]) {
                    cart_out.at<cv::Vec3b>(point_list[i]) = cv::Vec3b(0, 255, 0);
                    cart_out2.at<cv::Vec3b>(point_list[i]) = cv::Vec3b(0, 255, 0);
                }
            drawStraightLine(cart_out, best_model, cv::Scalar(0, 0, 255));
            drawStraightLine(cart_out2, best_model, cv::Scalar(0, 0, 255));

            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_image", cart_image);
            cv::imshow("cart_filtered", cart_filtered);
            cv::imshow("cart_thresh", cart_thresh);
            cv::imshow("cart_out", cart_out);
            cv::imshow("cart_out2", cart_out2);
            cv::waitKey(10);
        }
        cv::waitKey(0);
    }

return 0;
}
