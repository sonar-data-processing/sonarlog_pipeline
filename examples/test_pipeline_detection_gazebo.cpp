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
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_util/Converter.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace sonar_processing;
using namespace sonar_processing::denoising;

inline void loadSonarHolder(const base::samples::Sonar& sample, sonar_processing::SonarHolder& sonar_holder) {
    sonar_holder.Reset(sample.bins,
        rock_util::Utilities::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count);
}

cv::Rect getMaskLimits(const cv::Mat& mask) {
    // check if mask is valid
    size_t mask_pixels = cv::countNonZero(mask);
    if(!mask_pixels) return cv::Rect();

    // find mask limits
    cv::Mat points;
    cv::findNonZero(mask, points);
    return cv::boundingRect(points);
}

bool pipeline_detection(const cv::Mat& src, const cv::Mat& mask, std::pair<cv::Point2f, cv::Point2f>& best_model, double fit_rate, float angle, int step_angle, int step_scan, int neighborhood = 0) {
    bool found = false;
    double best_score = 1;

    // find mask limits
    cv::Rect rect = getMaskLimits(mask);

    double f = (rect.width > rect.height) ? 0.9 : 0.5;
    size_t min_valid_pixels = (rect.height) * f * (neighborhood * 2 + 1);

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
                if(p1.x > rect.br().x && p2.x > rect.br().x)
                    break;
                else
                    continue;
            }

            // get all points on the line segment using a 8-connected pixels
            cv::LineIterator line_points(src, p1, p2);

            // calculate the proportional average
            double accum_sum = 0;
            int valid_pixels = 0;
            for (int k = 0; k < line_points.count; k++, ++line_points) {
                cv::Point point = line_points.pos();
                for (int l = -neighborhood; l <= neighborhood; l++) {
                    cv::Point current_point = cv::Point(point.x + l, point.y);
                    accum_sum += src.at<float>(current_point);
                    if(mask.at<uchar>(current_point))
                        valid_pixels++;
                }
            }

            std::cout << "Valid pixels: " << valid_pixels << std::endl;
            std::cout << "Min Valid pixels: " << min_valid_pixels << std::endl;

            // if there are valid pixels, evaluate the results
            if (valid_pixels < min_valid_pixels)
                continue;
            double avg = accum_sum / valid_pixels;

            if ((avg > fit_rate) || (valid_pixels < min_valid_pixels) || avg > best_score)
                continue;

            // store the best results
            cv::clipLine(rect, p1, p2);
            best_model = std::make_pair(cv::Point2f(p1.x, p1.y), cv::Point2f(p2.x, p2.y));
            best_score = avg;
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

bool isBadQuality(const cv::Mat& src, size_t m_cols, size_t n_rows, double threshold = 0.35) {
    for (size_t j = 0; j < n_rows; j++) {
        for (size_t i = 0; i < m_cols; i++) {
            cv::Mat block = src(cv::Rect(i + src.cols / m_cols, j * src.rows / n_rows, src.cols / m_cols, src.rows / n_rows));
            double avg = cv::sum(block)[0] / block.total();
            if (avg < threshold) return true;
        }
    }
    return false;
}

void writeImageToFile(const cv::Mat& frame, const std::string& filename) {
    cv::FileStorage storage(filename, cv::FileStorage::WRITE);
    storage << "insonification_pattern" << frame;
    storage.release();
}

cv::Mat readImageFromFile(const std::string& filename) {
    cv::FileStorage storage(filename, cv::FileStorage::READ);
    cv::Mat frame;
    storage["insonification_pattern"] >> frame;
    storage.release();
    return frame;
}

cv::Mat getSymetricData(const cv::Mat& src) {
    cv::Mat left  = src(cv::Rect(0, 0, src.cols * 0.5, src.rows));
    cv::Mat right = src(cv::Rect(src.cols * 0.5, 0, src.cols * 0.5, src.rows));

    cv::Mat left_mirror;
    cv::flip(left, left_mirror, 1);

    cv::Mat out_right = 1 - (left_mirror + right);
    cv::medianBlur(out_right, out_right, 3);
    out_right.setTo(0, out_right < 0.8);

    cv::Mat out_left;
    cv::flip(out_right, out_left, 1);

    cv::Mat dst;
    cv::hconcat(out_left, out_right, dst);
    return dst;
}

bool intersectSymetricData(const cv::Mat& src, const std::pair<cv::Point2f, cv::Point2f>& best_model, cv::Rect rect, double threshold = 0.2) {
    cv::Mat symetric = getSymetricData(src);

    cv::Point2f p1 = cv::Point2f(best_model.first.x - rect.x, best_model.first.y - rect.y);
    cv::Point2f p2 = cv::Point2f(best_model.second.x - rect.x, best_model.second.y - rect.y);
    cv::LineIterator line_points(src, p1, p2);

    int total_pixels = 0;
    int crossed_pixels = 0;

    for (; total_pixels < line_points.count; total_pixels++, ++line_points) {
        cv::Point current_point = line_points.pos();
        if(symetric.at<float>(current_point))
            crossed_pixels++;
    }

    double percentage = crossed_pixels * 1.0 / total_pixels;
    return (percentage > threshold);
}

bool isBadDepth(const cv::Mat& mask, double range, double percentage = 0.30) {
    double sonar_resolution = range / mask.cols;
    cv::Point center(mask.cols * 0.5, mask.rows - 1);
    int idx = center.y;

    for (size_t i = 0; i < mask.rows; i++) {
        if(mask.at<uchar>(cv::Point(center.x, center.y - i)) == 255) {
            idx = i;
            break;
        }
    }

    double resolution = range / mask.rows;
    double distance = idx * resolution;
    return (distance > (range * percentage));
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        // DATA_PATH_STRING + "/logs/pipeline/simulation.1.log",
        "/home/romulo/workspace/sonar_toolkit/rock_util/scripts/output.0.log"
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(2);
    sonar_processing::SonarHolder sonar_holder;
    base::samples::Sonar sample;
    size_t start_index = (argc == 2) ? atoi(argv[1]) : 0;
    double scale_factor = 0.7;
    double range = 17;
    cv::Mat bkgd_pattern = cv::Mat();
    double bkgd_factor = 0.5;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("sonar_multibeam_imager.sonar_samples");
        stream.set_current_sample_index(start_index);
        bool scan_vertical = true;

        while (stream.current_sample_index() < stream.total_samples()) {
            clock_t tStart = clock();

            stream.next<base::samples::Sonar>(sample);
            loadSonarHolder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), scale_factor, scale_factor);

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw * 20);
            if(rls.getBuffer_size() < rls.getWindow_size()) continue;

            /* drawable area */
            cv::Mat cart_mask = sonar_holder.cart_image_mask();
            cv::resize(cart_mask, cart_mask, cart_raw.size());

            /* cartesian roi image */
            cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_denoised, cart_mask, 0.1);
            cv::Rect rect_roi = getMaskLimits(cart_roi);
            if(!rect_roi.area()) continue;

            /* insonification pattern */
            cv::Mat cart_corrected = cart_denoised.clone();
            if(!bkgd_pattern.empty()) {
                cart_corrected -= (bkgd_pattern * bkgd_factor);
                cart_corrected.setTo(0, cart_corrected < 0);
            }

            /* filtering */
            cv::Mat cart_aux;
            cart_corrected(rect_roi).convertTo(cart_aux, CV_8U, 255);
            // preprocessing::adaptive_clahe(cart_aux, cart_aux, 5);
            cart_aux.convertTo(cart_aux, CV_32F, 1.0 / 255);
            cv::Mat cart_filtered = cv::Mat::zeros(cart_corrected.size(), CV_32F);
            cart_aux.copyTo(cart_filtered(rect_roi));
            cv::multiply(cart_filtered, cart_raw, cart_filtered);

            /* skip bad frames */
            bool bad_quality = isBadQuality(cart_filtered(rect_roi), 3, 3);
            bool bad_depth = isBadDepth(cart_roi, range);

            /* output results */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_mask", cart_mask);
            cv::imshow("cart_roi", cart_roi);
            cv::imshow("cart_corrected", cart_corrected);
            cv::imshow("cart_filtered", cart_filtered);

            cv::Mat cart_out;
            cv::cvtColor(cart_raw, cart_out, CV_GRAY2BGR);
            cart_out.convertTo(cart_out, CV_8UC3, 255);

            /* pipeline detection */
            bool found = false;
            if(!bad_quality && !bad_depth) {
                std::pair<cv::Point2f, cv::Point2f> best_model;
                /* scan for a vertical pipeline */
                if(scan_vertical){
                    found = pipeline_detection(cart_filtered, cart_roi, best_model, 0.2, 45, 5, 1);
                    if (found) {
                        found = !intersectSymetricData(cart_filtered(rect_roi), best_model, rect_roi);
                    }

                    std::cout << "Vertical scanning... Found? " << found << std::endl;
                    scan_vertical = found;
                }
                /* scan for a horizontal pipeline */
                else {
                    found = pipeline_detection(cart_filtered.t(), cart_roi.t(), best_model, 0.2, 45, 5, 1);
                    best_model.first = cv::Point2f(best_model.first.y, best_model.first.x);
                    best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
                    if(found) found = !intersectSymetricData(cart_filtered(rect_roi), best_model, rect_roi);
                    std::cout << "Horizontal scanning... Found? " << found << std::endl;
                    scan_vertical = !found;
                }

                /* if a candidate target is found, check if it is contained in a symetric noise with low intensities */
                if (found) {
                    // if(!scan_vertical) {
                    //     best_model.first = cv::Point2f(best_model.first.y, best_model.first.x);
                    //     best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
                    // }
                    //
                    // found = !intersectSymetricData(cart_filtered(rect_roi), best_model, rect_roi);
                    //
                    // if(found) {
                        cv::line(cart_out, best_model.first, best_model.second, cv::Scalar(0, 255, 0), 2, CV_AA);
                        cv::line(cart_out, cv::Point(0, cart_out.rows * 0.5), cv::Point(cart_out.cols - 1, cart_out.rows * 0.5), cv::Scalar(0, 0, 255), 1, CV_AA);
                        cv::line(cart_out, cv::Point(cart_out.cols * 0.5, 0), cv::Point(cart_out.cols * 0.5, cart_out.rows - 1), cv::Scalar(0, 0, 255), 1, CV_AA);
                    }
                }
            }

            // std::cout << "Vertical: " << scan_vertical << ", Horizontal: " << !scan_vertical << std::endl;
            std::cout << "Bad Frame? " << bad_quality << "," << bad_depth << std::endl;

            cv::imshow("cart_out", cart_out);
            cv::waitKey();
            clock_t tEnd = clock();
            double elapsed_secs = double (tEnd - tStart) / CLOCKS_PER_SEC;
            std::cout << " ==================== FPS: " << (1.0 / elapsed_secs) << std::endl;
            std::cout << " ==================== IDX: " << stream.current_sample_index() << std::endl;
        }

        cv::waitKey(0);
    }

return 0;
}
