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
#include "sonar_processing/Denoising.hpp"

using namespace sonarlog_pipeline;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

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

double getLineMean(const cv::Mat& src, cv::Point p1, cv::Point p2, const LineScanConfig& config, uint min_valid_pixels) {
    // get all points on the line segment using a 8-connected pixels
    cv::LineIterator line_points(src, p1, p2);
    cv::Rect rect(0, 0, src.size().width, src.size().height);

    // calculate the line intensity mean
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

    if (valid_pixels < min_valid_pixels) return 1;
    else return (sum / valid_pixels);
}

double getLineStddev(const cv::Mat& src, cv::Point p1, cv::Point p2, const LineScanConfig& config, uint min_valid_pixels, double mean) {
    // get all points on the line segment using a 8-connected pixels
    cv::LineIterator line_points(src, p1, p2);
    cv::Rect rect(0, 0, src.size().width, src.size().height);

    // calculate the line intensity stddev
    double accum = 0;
    uint valid_pixels = 0;
    for (int k = 0; k < line_points.count; k++, ++line_points) {
         cv::Point point = line_points.pos();
         for (int l = -config.neighborhood; l <= config.neighborhood; l++) {
             cv::Point current_point = cv::Point(point.x + l, point.y);
             if(!rect.contains(current_point)) continue;
             double value = src.at<float>(current_point);
             accum += (value - mean) * (value - mean);
             valid_pixels++;
         }
    }
    if (valid_pixels < min_valid_pixels) return 1;
    else return std::sqrt(accum / (double) valid_pixels);
}

cv::Mat getSymmetricInterferences(const cv::Mat& src, const cv::Mat& mask, size_t step = 1) {
    cv::Rect rect = getMaskLimits(mask);
    cv::Point center(mask.cols * 0.5, mask.rows - 1);
    int top_left = cv::norm(center - rect.tl());
    int offset_y = mask.rows - rect.br().y;

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8U);

    for (size_t i = offset_y; i < top_left; i += step) {
        // draw ellipse
        cv::Mat aux = cv::Mat::zeros(src.size(), CV_8U);
        cv::Point center_aux(aux.cols * 0.5, aux.rows - 1);
        cv::Point pt(center_aux.x, center_aux.y + offset_y);
        cv::ellipse(aux, pt, cv::Size(i, i), 180, 0, 180, cv::Scalar(255));

        // get all points from this intersection
        std::vector<cv::Point> points;
        cv::findNonZero(aux, points);

        // calculate the ellipse mean
        double sum = 0;
        for (size_t i = 0; i < points.size(); i++)
            sum += src.at<float>(points[i].y, points[i].x);
        double mean = sum / points.size();

        // calculate the ellipse stddev
        double accum = 0;
        for (size_t i = 0; i < points.size(); i++) {
            double value = src.at<float>(points[i]);
            accum += (value - mean) * (value - mean);
        }
        double stddev = std::sqrt(accum / points.size());

        if (mean > 0.30 || stddev > 0.25) continue;
        for (size_t i = 0; i < points.size(); i++)
            dst.at<uchar>(points[i]) = 255;
    }
    return dst;
}

bool intersectSymmetricInterferences(const cv::Mat& src, cv::Point p1, cv::Point p2, float threshold = 0.15) {
    cv::LineIterator line_points(src, p1, p2);

    cv::Mat aux = cv::Mat::zeros(src.size(), CV_8U);
    cv::line(aux, p1, p2, cv::Scalar(255), 1, CV_AA);
    size_t total_pixels = cv::countNonZero(aux);

    cv::Mat crossed = aux & src;
    size_t crossed_pixels = cv::countNonZero(crossed);

    float percentage = crossed_pixels * 1.0 / total_pixels;
    return (percentage > threshold);
}

bool isValidNeighborhood(const cv::Mat& src, cv::Point p1, cv::Point p2, const LineScanConfig& config, size_t min_valid_pixels) {
    double max_stddev = 0.20;

    size_t pxs_left = 0;
    size_t pxs_right = 0;
    bool reached_left = false;
    bool reached_right = false;

    std::cout << "-=-=-=-=-=-=-=-=" << std::endl;

    // cv::Mat src = src2.clone();
    // cv::Rect rect(p1.x - 20, 0, 40, src2.rows);
    // src(rect).setTo(0);

    for (size_t i = 1; i < src.cols; i++) {
        // cv::Mat temp = src2.clone();
        // cv::cvtColor(src, temp, cv::COLOR_GRAY2BGR);
        // cv::line(temp, p1, p2, cv::Scalar(0,0,255), 1, CV_AA);

        if (!reached_left) {
            cv::Point p1_left(p1.x - i, p1.y), p2_left(p2.x - i, p2.y);
            double mean = getLineMean(src, p1_left, p2_left, config, min_valid_pixels);
            double stddev = getLineStddev(src, p1_left, p2_left, config, min_valid_pixels, mean);
            if (mean > config.fit_rate || stddev > max_stddev)
                reached_left = true;
            else pxs_left++;
            // cv::line(temp, p1_left, p2_left, cv::Scalar(0,255,0), 1, CV_AA);

            // std::cout << "Left: " << mean << "," << stddev << "," << pxs_left << "," << reached_left << std::endl;
        }

        if (!reached_right) {
            cv::Point p1_right(p1.x + i, p1.y), p2_right(p2.x + i, p2.y);
            double mean = getLineMean(src, p1_right, p2_right, config, min_valid_pixels);
            double stddev = getLineStddev(src, p1_right, p2_right, config, min_valid_pixels, mean);
            if (mean > config.fit_rate || stddev > max_stddev)
                reached_right = true;
            else pxs_right++;
            // cv::line(temp, p1_right, p2_right, cv::Scalar(0,255,0), 1, CV_AA);

            // std::cout << "Right: " << mean << "," << stddev << "," << pxs_right << "," << reached_right << std::endl;
        }

        // std::cout << "Pxs - Left: " << pxs_left << std::endl;
        // std::cout << "Pxs - Right: " << pxs_right << std::endl;
        // cv::imshow("temp", temp);
        // cv::waitKey();

        if (reached_left && reached_right) break;
    }
    // std::cout << "Total: " << (pxs_left + pxs_right) << std::endl;

    if ((pxs_left + pxs_right) > 10) return false;
    return true;
}

bool isCornerLine(const cv::Mat& src, cv::Point p1, cv::Point p2, const LineScanConfig& config) {
    cv::Mat img_line = cv::Mat::zeros(src.size(), CV_8U);
    cv::line(img_line, p1, p2, cv::Scalar::all(255), 1, CV_AA);

    cv::Mat img_corners = cv::Mat::zeros(src.size(), CV_8U);
    cv::Rect left_corner(0, 0, src.cols * 0.05, src.rows - 1);
    cv::Rect right_corner(src.cols * 0.95, 0, src.cols * 0.05, src.rows - 1);
    img_corners(left_corner).setTo(255);
    img_corners(right_corner).setTo(255);

    cv::Mat crossed = img_line & img_corners;

    int total_pixels   = cv::countNonZero(img_line);
    int crossed_pixels = cv::countNonZero(crossed);

    if(!total_pixels) return true;
    float percentage = crossed_pixels * 1.0 / total_pixels;
    return (percentage > 0.5);
}


bool isValidCandidate(const cv::Mat& src, const cv::Mat& symmetric, cv::Point p1, cv::Point p2, const LineScanConfig& config, size_t min_valid_pixels) {
    // check intersection in symetric images
    if (intersectSymmetricInterferences(symmetric, p1, p2)) return false;

    // avoid line closer to image corners
    if (isCornerLine(src, p1, p2, config)) return false;

    // avoid line in black regions
    if(!isValidNeighborhood(src, p1, p2, config, min_valid_pixels)) return false;

    return true;
}

bool pipeline_detection(const cv::Mat& src, const cv::Mat& mask, std::pair<cv::Point2f, cv::Point2f>& best_model, const LineScanConfig& config) {
    bool found = false;
    double best_score = 1;
    double max_stddev = 0.20;

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

    cv::Mat symmetric;

    for (int i = (center.x - angle_range); i < (center.x + angle_range); i += config.step_angle) {
        for (int j = -src.cols; j < src.cols; j += config.step_scan) {
            cv::Point p1(i + j, rect.y);
            cv::Point p2(center.x + j, rect.y + rect.height - 1);

            // check if the reference points p1 and p2 are contained within the mask image
            if(!rect.contains(p1) && !rect.contains(p2)) {
                if(p1.x > rect.br().x && p2.x > rect.br().x) break;
                else continue;
            }

            cv::clipLine(rect, p1, p2);

            // mean
            double mean = getLineMean(src, p1, p2, config, min_valid_pixels);
            if ((mean > config.fit_rate) || (mean > best_score)) continue;

            // stddev
            double stddev = getLineStddev(src, p1, p2, config, min_valid_pixels, mean);
            if (stddev > max_stddev) continue;

            std::cout << "Mean: " << mean << std::endl;
            std::cout << "Stddev: " << stddev << std::endl;

            if (symmetric.empty()) symmetric = getSymmetricInterferences(src, mask);

            bool isValid = isValidCandidate(src, symmetric, p1, p2, config, min_valid_pixels);
            std::cout << "isValid() = " << isValid << std::endl;

            if (!isValid) continue;

            // cv::Mat aux = out.clone();
            // cv::line(aux, p1, p2, cv::Scalar(0,0,255), 1, CV_AA);
            // cv::line(aux, cv::Point(p1.x - 10, p1.y), cv::Point(p2.x - 10, p2.y), cv::Scalar(255, 0, 0), 1, CV_AA);
            // cv::line(aux, cv::Point(p1.x + 10, p1.y), cv::Point(p2.x + 10, p2.y), cv::Scalar(255, 0, 0), 1, CV_AA);
            // cv::imshow("aux", aux);
            // cv::waitKey();

            // store the best results
            best_model = std::make_pair(cv::Point2f(p1.x, p1.y), cv::Point2f(p2.x, p2.y));
            best_score = mean;
            found = true;
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

bool pipelineScanner(const cv::Mat& src, const cv::Mat& mask, bool &start_vertical, std::pair<cv::Point2f, cv::Point2f>& best_model) {
    LineScanConfig vrt(0.18, base::Angle::fromDeg(75), 5, 2, 0);
    LineScanConfig hrz(0.18, base::Angle::fromDeg(30), 5, 5, 0);

    bool found = false;

    // starting scan vertically, then horizontally
    if(start_vertical) {
        // scan vertically
        found = pipeline_detection(src, mask, best_model, vrt);

        // scan horizontally
        if(!found) {
            found = pipeline_detection(src.t(), mask.t(), best_model, hrz);
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
        found = pipeline_detection(src.t(), mask.t(), best_model, vrt);
        if (found) {
            best_model.first  = cv::Point2f(best_model.first.y, best_model.first.x);
            best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
        }

        // scan vertically
        else {
            found = pipeline_detection(src, mask, best_model, hrz);
            if (found) start_vertical = true;
        }
    }

    return found;
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        // "/arquivos/log_pipeline/20171114-1156/gemini_20171114-1156.0.log"
        "/arquivos/log_pipeline/20171115-0907/gemini_20171115-0907.0.log"
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);

    sonar_processing::SonarHolder2 sonar_holder;
    base::samples::Sonar sample;
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;
    unsigned int rls_size = 4;
    cv::Size max_size(710,410);

    for (uint32_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");
        // rock_util::LogStream stream = reader.stream("sonar_multibeam_imager.sonar_samples");

        stream.set_current_sample_index(start_index);
        bool start_vertical = true;
        RLS rls(rls_size);

        while (stream.current_sample_index() < stream.total_samples()) {
            std::cout << " ==================== IDX: " << stream.current_sample_index() << std::endl;

            clock_t tStart = clock();
            stream.next<base::samples::Sonar>(sample);
            sonar_holder.update(sample);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.getCartImage();
            if (cart_raw.size().height > max_size.height)
                cv::resize(cart_raw, cart_raw, max_size);

            /* denoising process */
            cv::Mat cart_denoised;
            if (rls_size > 1) {
                cart_denoised = rls.sliding_window(cart_raw);
                if(rls.getBuffer_size() < rls.getWindow_size()) continue;
            } else
                cart_raw.copyTo(cart_denoised);

            /* drawable area */
            cv::Mat cart_mask = sonar_holder.getCartImageMask();
            cv::resize(cart_mask, cart_mask, cart_raw.size());

            /* cartesian roi image */
            cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_denoised, cart_mask, 0.25);
            cv::Rect rect_roi = getMaskLimits(cart_roi);

            /* filtering */
            cv::Mat cart_aux, cart_filtered;
            cart_denoised(rect_roi).copyTo(cart_filtered);
            cart_filtered = insonification_correction(cart_filtered);
            cart_filtered.convertTo(cart_aux, CV_8U, 255);
            cart_aux = preprocessing::adaptiveClahe(cart_aux, 5, cv::Size(4,4));
            cart_aux.convertTo(cart_filtered, CV_32F, 1.0 / 255);

            bool bad_rect = ((rect_roi.size().width < (cart_denoised.cols * 0.3)) || (rect_roi.size().width < (rect_roi.size().height * 0.9)));
            bool bad_quality = isBadQuality(cart_denoised(rect_roi), 4, 4, 0.1);
            bool bad_depth = (rect_roi.br().y < (cart_denoised.rows * 0.65));

            if (bad_rect || bad_quality || bad_depth) {
                std::string s;
                if (bad_rect)    s = s + "| Bad Rect";
                if (bad_quality) s = s + "| Bad Quality";
                if (bad_depth)   s = s + "| Bad Depth";
                std::cout << s << std::endl;
            }

            cv::Mat cart_out;
            cart_denoised.convertTo(cart_out, CV_8U, 255);
            cv::cvtColor(cart_out, cart_out, cv::COLOR_GRAY2BGR);

            // // /* pipeline detection */
            if (!bad_quality && !bad_depth && !bad_rect) {
                cv::rectangle(cart_out, rect_roi, cv::Scalar(0,255,255));
                cv::putText(cart_out, "Searching Area", cv::Point(rect_roi.x, rect_roi.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,255), 1, CV_AA);
                // cv::line(cart_out, cv::Point(cart_out.cols * 0.5, 0), cv::Point(cart_out.cols * 0.5, cart_out.rows * 0.65), cv::Scalar(0,0,255), 2, CV_AA);
                std::pair<cv::Point2f, cv::Point2f> best_model;
                bool found = pipelineScanner(cart_filtered, cart_roi, start_vertical, best_model);
                if (found) {
                    best_model.first  = cv::Point2f(best_model.first.x + rect_roi.x, best_model.first.y + rect_roi.y);
                    best_model.second = cv::Point2f(best_model.second.x + rect_roi.x, best_model.second.y + rect_roi.y);
                    cv::line(cart_out, best_model.first, best_model.second, cv::Scalar(0,255,0), 2, CV_AA);
                    cv::putText(cart_out, "Pipeline found!", cv::Point(10, cart_out.rows - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,0), 1, CV_AA);

            //         // // convert to world points
            //         // double max_range  = sample.getBinStartDistance(sample.bin_count);
            //         // base::Vector2d p1 = getWorldPoint(best_model.first, cart_out.size(), max_range);
            //         // base::Vector2d p2 = getWorldPoint(best_model.second, cart_out.size(), max_range);
                }
            }


            /* output results */
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
