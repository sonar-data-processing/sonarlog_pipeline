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
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/SonarHolder2.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_pipeline/Application.hpp"

using namespace sonarlog_pipeline;
using namespace sonar_processing;

struct LineScanConfig {
    float fit_rate;
    base::Angle angle;
    uint step_angle;
    uint step_scan;
    uint neighborhood;

    LineScanConfig()
        : fit_rate(0.20)
        , angle(base::Angle::fromDeg(60))
        , step_angle(5)
        , step_scan(5)
        , neighborhood(0)
    {};

    LineScanConfig(float fit_rate, base::Angle angle, int step_angle, int step_scan, int neighborhood)
        : fit_rate(fit_rate)
        , angle(angle)
        , step_angle(step_angle)
        , step_scan(step_angle)
        , neighborhood(neighborhood)
    {};
};

cv::Rect getMaskLimits(const cv::Mat& mask) {
    // check if mask is valid
    size_t mask_pixels = cv::countNonZero(mask);
    if(!mask_pixels) return cv::Rect();

    // find mask limits
    cv::Mat points;
    cv::findNonZero(mask, points);
    return cv::boundingRect(points);
}

cv::Mat insonification_correction (const cv::Mat& src) {
    cv::Mat dst = src.clone();

    // calculate the proportional mean of each image row
    std::vector<double> row_mean(src.rows, 0);
    for (size_t i = 0; i < src.rows; i++) {
        unsigned int num_points = src.cols;
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

bool pipeline_detection(const cv::Mat& src, std::pair<cv::Point2f, cv::Point2f>& best_model, const LineScanConfig& config) {
    bool found = false;
    double best_score = 1;
    double max_stddev = 0.15;

    // find mask limits
    cv::Rect rect(0, 0, src.cols, src.rows);

    double f = (rect.width > rect.height) ? 0.9 : 0.5;
    size_t min_valid_pixels = (rect.height) * f * (config.neighborhood * 2 + 1);

    // pipeline searching using a sliding line segment
    cv::Point2f center(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    int angle_range = tan(config.angle.rad) * rect.height;

    cv::Mat out;
    src.convertTo(out, CV_8U, 255);
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

    for (int i = (center.x - angle_range); i < (center.x + angle_range); i += config.step_angle) {
        for (int j = -src.cols; j < src.cols; j += config.step_scan) {
            cv::Point p1(i + j, rect.y);
            cv::Point p2(center.x + j, rect.y + rect.height - 1);

            // check if the reference points p1 and p2 are contained within the mask image
            if(!rect.contains(p1) && !rect.contains(p2)) {
                if(p1.x > rect.br().x && p2.x > rect.br().x)
                    break;
                else
                    continue;
            }


            // get all points on the line segment using a 8-connected pixels
            cv::clipLine(rect, p1, p2);
            cv::LineIterator line_points(src, p1, p2);

            // cv::Mat aux = out.clone();
            // cv::line(aux, p1, p2, cv::Scalar(0,0,255), 1);
            // cv::imshow("aux", aux);
            // cv::waitKey();

            // calculate the proportional average
            double sum = 0;
            uint valid_pixels = 0;
            for (int k = 0; k < line_points.count; k++, ++line_points) {
                cv::Point point = line_points.pos();
                for (int l = -config.neighborhood; l <= config.neighborhood; l++) {
                    cv::Point current_point = cv::Point(point.x + l, point.y);
                    if(!rect.contains(current_point)) continue;
                    sum += src.at<float>(current_point);
                    valid_pixels++;
                }
            }

            // if there are valid pixels, evaluate the results (mean and stddev)
            if (valid_pixels < min_valid_pixels)
                continue;

            // mean
            double mean = sum / valid_pixels;

            if ((mean > config.fit_rate) || (valid_pixels < min_valid_pixels) || (mean > best_score))
                continue;

            // stddev
            line_points = cv::LineIterator(src, p1, p2);
            double accum = 0;
            for (int k = 0; k < line_points.count; k++, ++line_points) {
                 cv::Point point = line_points.pos();
                 for (int l = -config.neighborhood; l <= config.neighborhood; l++) {
                     cv::Point current_point = cv::Point(point.x + l, point.y);
                     if(!rect.contains(current_point)) continue;
                     double value = src.at<float>(current_point);
                     accum += (value - mean) * (value - mean);
                 }
            }
            double stddev = std::sqrt(accum / (double) valid_pixels);

            if (stddev > max_stddev) continue;

            // store the best results
            best_model = std::make_pair(cv::Point2f(p1.x, p1.y), cv::Point2f(p2.x, p2.y));
            best_score = mean;
            found = true;

    //         // std::cout << "Mean: " << mean << std::endl;
    //         // std::cout << "Stddev: " << stddev << std::endl;
    //
        }
    }
    return found;
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

bool isBadQuality(const cv::Mat& src, size_t m_cols, size_t n_rows, double threshold = 0.15) {
    // cv::imshow("src", src);
    // cv::waitKey();

    if (src.size().area() < (m_cols * n_rows * 3))
        return true;

    for (size_t j = 0; j < n_rows; j++) {
        for (size_t i = 0; i < m_cols; i++) {
            cv::Mat block = src(cv::Rect(i + src.cols / m_cols, j * src.rows / n_rows, src.cols / m_cols, src.rows / n_rows));
            double avg = cv::sum(block)[0] / block.total();
            if (avg < threshold) return true;
        }
    }
    return false;
}

bool pipeline_scanner(const cv::Mat& src, bool &start_vertical, std::pair<cv::Point2f, cv::Point2f>& best_model) {
    LineScanConfig vrt(0.20, base::Angle::fromDeg(75), 5, 2, 0);
    LineScanConfig hrz(0.20, base::Angle::fromDeg(30), 5, 5, 0);

    bool found = false;

    // starting scan vertically, then horizontally
    if(start_vertical) {
        // scan vertically
        found = pipeline_detection(src, best_model, vrt);

        // scan horizontally
        if(!found) {
            found = pipeline_detection(src.t(), best_model, hrz);
            if (found) {
                best_model.first  = cv::Point2f(best_model.first.y, best_model.first.x);
                best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
                start_vertical = false;
            }
        }
    }

    // if the pipeline was not found, scan horizontally
    else {
        // scan horizontally
        found = pipeline_detection(src.t(), best_model, vrt);
        if (found) {
            best_model.first  = cv::Point2f(best_model.first.y, best_model.first.x);
            best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
        }

        // scan vertically
        else {
            found = pipeline_detection(src, best_model, hrz);
            if (found) start_vertical = true;
        }
    }

    return found;
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        // "/arquivos/Logs/gemini/gemini_pipeline_logs/demo_april_20170427/testday.0.log",
        // "/arquivos/Logs/gemini/gemini_pipeline_logs/demo_april_20170427/testday.1.log",
        // "/arquivos/Logs/gemini/gemini_pipeline_logs/demo_april_20170427/testday.2.log",
        // "/arquivos/Logs/gemini/gemini_pipeline_logs/demo_april_20170427/testday.3.log",
        // DATA_PATH_STRING + "/logs/pipeline/testsite.0.log",
        // DATA_PATH_STRING + "/logs/pipeline/testsite.1.log",
        // "/home/romulo/workspace/sonar_toolkit/rock_util/scripts/pipe.0.log",
        // "/home/romulo/workspace/sonar_toolkit/rock_util/scripts/pipe.1.log",
        // DATA_PATH_STRING + "/logs/pipeline/nodata.0.log",
        // DATA_PATH_STRING + "/logs/pipeline/simulation.4.log",
        // "/home/romulo/Desktop/sim_logs/out.log",
        "/home/romulo/workspace/sonar_toolkit/rock_util/scripts/out2.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);

    sonar_processing::SonarHolder2 sonar_holder;
    base::samples::Sonar sample;
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;
    double scale_factor = 0.4;
    bool binarization = false;

    for (uint32_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");
        // rock_util::LogStream stream = reader.stream("sonar_multibeam_imager.sonar_samples");

        stream.set_current_sample_index(start_index);
        bool start_vertical = true;

        while (stream.current_sample_index() < stream.total_samples()) {
            clock_t tStart = clock();

            stream.next<base::samples::Sonar>(sample);
            sonar_holder.update(sample);

            std::cout << " ==================== IDX: " << stream.current_sample_index() << std::endl;

            /* current frame */
            cv::Mat cart_raw = sonar_holder.getCartImage();
            cv::resize(cart_raw, cart_raw, cv::Size(), scale_factor, scale_factor);

            /* drawable area */
            cv::Mat cart_mask = sonar_holder.getCartImageMask();
            cv::resize(cart_mask, cart_mask, cart_raw.size());

            /* cartesian roi image */
            cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_raw, cart_mask, 0.2);
            cv::Rect rect_roi = getMaskLimits(cart_roi);
            if ((rect_roi.size().width < (0.30 * cart_raw.cols)) || (rect_roi.size().height > (0.60 * cart_raw.rows)))
                continue;

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            if (binarization)
                cv::Mat(cart_raw > 0)(rect_roi).copyTo(cart_aux);
            else
                cart_raw(rect_roi).copyTo(cart_aux);
            cart_filtered = insonification_correction(cart_aux);
            cart_filtered.convertTo(cart_aux, CV_8U, 255);
            cart_aux = preprocessing::adaptiveClahe(cart_aux, 5, cv::Size(4,4));
            cart_aux.convertTo(cart_filtered, CV_32F, 1.0 / 255);

            /* skip bad frames */
            bool bad_quality = isBadQuality(cart_raw(rect_roi), 3, 3);
            bool bad_depth = (rect_roi.br().y < (cart_raw.rows * 0.5));
            std::cout << "Bad depth: " << rect_roi.br().y << "," << (cart_raw.rows * 0.6) << std::endl;
            std::cout << "Bad quality: " << bad_quality << std::endl;

            cv::Mat cart_out;
            cart_raw.convertTo(cart_out, CV_8U, 255);
            cv::cvtColor(cart_out, cart_out, cv::COLOR_GRAY2BGR);

            /* pipeline detection */
            std::pair<cv::Point2f, cv::Point2f> best_model;
            if (bad_quality || bad_depth) {
                cv::putText(cart_out, "Bad frame", cv::Point(10, cart_out.rows - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,0,255), 1, CV_AA);
            } else {
                cv::rectangle(cart_out, rect_roi, cv::Scalar(0,255,255));
                cv::putText(cart_out, "Searching Area", cv::Point(rect_roi.x, rect_roi.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,255), 1, CV_AA);
                bool found = pipeline_scanner(cart_filtered, start_vertical, best_model);
                if (found) {
                    best_model.first  = cv::Point2f(best_model.first.x + rect_roi.x, best_model.first.y + rect_roi.y);
                    best_model.second = cv::Point2f(best_model.second.x + rect_roi.x, best_model.second.y + rect_roi.y);
                    cv::line(cart_out, best_model.first, best_model.second, cv::Scalar(0,255,0), 2, CV_AA);
                    cv::putText(cart_out, "Pipeline found!", cv::Point(10, cart_out.rows - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,0), 1, CV_AA);

                    // // convert to world points
                    // double max_range  = sample.getBinStartDistance(sample.bin_count);
                    // base::Vector2d p1 = getWorldPoint(best_model.first, cart_out.size(), max_range);
                    // base::Vector2d p2 = getWorldPoint(best_model.second, cart_out.size(), max_range);
                }
            }

            // /* output results */
            // cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_filtered", cart_filtered);
            cv::imshow("cart_out", cart_out);
            cv::waitKey(5);
            clock_t tEnd = clock();
            double elapsed_secs = double (tEnd - tStart) / CLOCKS_PER_SEC;


            std::cout << " ==================== FPS: " << (1.0 / elapsed_secs) << std::endl;

        }
    }

    return 0;
}
