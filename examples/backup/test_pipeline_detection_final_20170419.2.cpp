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

cv::Mat reference_histogram = (cv::Mat_<float>(21,1) <<    0.614824346, 0.6274731044,	0.6387303867, 0.6442708739,	0.6344197557,
                                                            0.5993996074,	0.5372821501, 0.4561557712,	0.3493434637, 0.2434227323,
                                                            0.2048770554, 0.242167915, 0.3245505577, 0.4345440575, 0.5130767172,
                                                            0.5603085825, 0.580626809, 0.5922801317, 0.5900253552,	0.5979811987, 0.6096264943);

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

bool double_check(const cv::Mat& src, const cv::Mat& mask, const std::pair<cv::Point2f, cv::Point2f>& best_model, int neighborhood = 10){
    bool found = false;

    // find mask limits
    cv::Rect rect = getMaskLimits(mask);

    double f = (rect.width > rect.height) ? 0.9 : 0.5;
    size_t min_valid_pixels = (rect.height) * f;
    std::vector<float> averages;

    // pipeline searching using a sliding line segment
    for (int j = -neighborhood; j <= neighborhood; j++) {
        cv::Point p1(best_model.first.x + j, best_model.first.y);
        cv::Point p2(best_model.second.x + j, best_model.second.y);

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
        if (valid_pixels < min_valid_pixels) {
            averages.push_back(0);
            continue;
        }
        double avg = accum_sum / valid_pixels;
        averages.push_back(avg);
    }

    cv::normalize(reference_histogram, reference_histogram, 0, 1, cv::NORM_MINMAX);

    cv::Mat current_histogram(averages);
    cv::Mat hist_mask = (current_histogram != 0);
    hist_mask.convertTo(hist_mask, CV_32F, 1.0 / 255);
    cv::multiply(reference_histogram, hist_mask, reference_histogram);

    cv::Mat hist_diff = reference_histogram - current_histogram;

    double min, max;
    cv::minMaxIdx(hist_diff, &min, &max, NULL, NULL);
    std::cout << "Hist_diff: " << hist_diff.t() << std::endl;
    std::cout << "Min/Max/Diff: " << min << "/" << max << "/" << (max - min) << std::endl;






    // std::cout << "Hist mask: " << hist_mask.t() << std::endl;
    //
    // cv::normalize(current_histogram, current_histogram, 0, 1, cv::NORM_MINMAX);

    // cv::Mat hist_diff = reference_histogram - current_histogram;

    // double dist = cv::compareHist(reference_histogram, current_histogram, CV_COMP_BHATTACHARYYA);
    // std::cout << "Distance Hist: " << dist << std::endl;
    // cv::Scalar mean, stddev;
    // cv::meanStdDev(hist_diff, mean, stddev);
    // std::cout << "Mean: " << mean[0] << ", Stddev: " << stddev[0] << std::endl;

    // int center_idx = averages.size() / 2;
    // cv::Rect left_side(0, 0, 1, averages.size() / 2);
    // cv::Rect right_side(0, averages.size() / 2 + 1, 1, averages.size() / 2);
    //
    // std::cout << " left_side " << left_side << " right_side " << right_side << std::endl;
    //
    // double max_left = 0, max_right = 0;
    // for (int i = center_idx + 2; i < averages.size(); i++) {
    //     if(averages[i] > averages[i + 1]) {
    //         max_right = averages[i];
    //         break;
    //     }
    // }
    //
    // for (int i = center_idx - 2; i > 0; i--) {
    //     if(averages[i] > averages[i - 1]) {
    //         max_left = averages[i];
    //         break;
    //     }
    // }
    //
    //
    // double left_diff = std::fabs(averages[center_idx] - max_left);
    // double right_diff = std::fabs(averages[center_idx] - max_right);
    //
    // std::cout << "Averages: " << cv::Mat(averages).t() << std::endl;
    // std::cout << "Center/LMax/RMax/LDiff/RDiff: " << averages[center_idx] << " / " << max_left << " / " << max_right << " / " << left_diff << " / " << right_diff << std::endl;
    return true;
    // return (result < 0.11);
}


cv::Mat insonificationCorrection (const cv::Mat& src) {
    cv::Mat dst = src.clone();

    // calculate the proportional mean of each image row
    std::vector<double> row_mean(src.rows, 0);
    for (size_t i = 0; i < src.rows; i++) {
        row_mean[i] = cv::sum(src.row(i))[0] / src.cols;
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

bool isBadFrame(const cv::Mat& frame, const cv::Mat& mask, size_t cols, size_t rows, double threshold = 0.35) {
    cv::Rect rect = getMaskLimits(mask);
    if(!rect.area()) return false;

    cv::imshow("frame", frame(rect));

    for (size_t j = 0; j < rows; j++) {
        for (size_t i = 0; i < cols; i++) {
            cv::Mat block = frame(cv::Rect(rect.x + i * rect.width / cols, rect.y + j * rect.height / rows, rect.width / cols, rect.height / rows));
            double avg = cv::sum(block)[0] / block.total();
            if (avg < threshold) return true;
            // std::cout << "avg: " << avg << std::endl;
            // cv::imshow("block", block);
            // cv::waitKey();
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

// cv::Mat avoidSymetricData(const cv::Mat& src) {
//     cv::Mat left  = src(cv::Rect(0, 0, src.cols * 0.5, src.rows));
//     cv::Mat right = src(cv::Rect(src.cols * 0.5, 0, src.cols * 0.5, src.rows));
//
//     cv::Mat left_mirror, out;
//
//     cv::flip(left, left_mirror, 1);
//     out = 1 - (left_mirror + 10 * right);
//     out.setTo(0, out < 0.5);
//     out.setTo(1, out >= 0.5);
//
//     cv::GaussianBlur(out, out, cv::Size(15, 15),0);
//     // cv::imshow("out", out);
//     // cv::waitKey();
//
//     right = right + out;
//     cv::flip(out, out, 1);
//     left = left + out;
//
//     cv::Mat dst;
//     cv::hconcat(left, right, dst);
//     return dst;
// }

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

bool intersectSymetricData(const cv::Mat& src, const std::pair<cv::Point2f, cv::Point2f>& best_model, cv::Rect rect, const cv::Point2f& ref, double threshold = 0.2) {
    cv::Mat symetric = getSymetricData(src(rect));
    cv::imshow("symetric", symetric);

    cv::Point2f p1 = best_model.first - ref;
    cv::Point2f p2 = best_model.second - ref;
    cv::LineIterator line_points(src, p1, p2);

    int total_pixels = 0;
    int crossed_pixels = 0;

    for (; total_pixels < line_points.count; total_pixels++, ++line_points) {
        cv::Point current_point = line_points.pos();
        if(symetric.at<float>(current_point))
            crossed_pixels++;
    }

    double percentage = crossed_pixels * 1.0 / total_pixels;
    std::cout << "Total/Crossed/Percentage: " << total_pixels << "/" << crossed_pixels << "/" << percentage << std::endl;
    return (percentage > threshold);
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
    double scale_factor = 0.4;
    double pattern_factor = 0.5;
    cv::Mat pattern = readImageFromFile("insonification_pattern.yml") * pattern_factor;


    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");
        stream.set_current_sample_index(start_index);
        bool scan_vertical = true;

        while (stream.current_sample_index() < stream.total_samples()) {

            clock_t tStart = clock();

            stream.next<base::samples::Sonar>(sample);
            loadSonarHolder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), scale_factor, scale_factor);

            /* drawable area */
            cv::Mat cart_mask = sonar_holder.cart_image_mask();
            cv::resize(cart_mask, cart_mask, cart_raw.size());

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);
            if(rls.getBuffer_size() < rls.getWindow_size()) continue;

            /* cartesian roi image */
            cv::Mat cart_roi = preprocessing::extract_cartesian_mask(cart_denoised, cart_mask, 0.25);
            cv::Rect rect_roi = getMaskLimits(cart_roi);

            /* insonification pattern */
            cv::Mat cart_corrected = cart_denoised - pattern;
            cart_corrected.setTo(0, cart_corrected < 0);

            /* filtering */
            cv::Mat cart_aux;
            cart_corrected(rect_roi).convertTo(cart_aux, CV_8U, 255);
            preprocessing::adaptive_clahe(cart_aux, cart_aux, 5);
            // cv::blur(cart_aux, cart_aux, cv::Size(3,3));
            cart_aux.convertTo(cart_aux, CV_32F, 1.0 / 255);
            // cv::multiply(cart_aux, cart_aux, cart_aux);

            /* discard symetric values on sonar image */
            cv::Mat cart_filtered = cv::Mat::zeros(cart_corrected.size(), CV_32F);
            cart_aux.copyTo(cart_filtered(rect_roi));

            /* skip bad frames */
            bool bad_frame = isBadFrame(cart_filtered, cart_roi, 3, 3);
            std::cout << "Bad frame? " << bad_frame << std::endl;

            /* output results */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_roi", cart_roi);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_corrected", cart_corrected);
            cv::imshow("cart_filtered", cart_filtered);

            cv::Mat cart_out;
            cv::cvtColor(cart_raw, cart_out, CV_GRAY2BGR);
            cart_out.convertTo(cart_out, CV_8UC3, 255);

            /* pipeline detection */
            if(!bad_frame) {
                std::pair<cv::Point2f, cv::Point2f> best_model;
                bool found = false;
                if(scan_vertical){
                    found = pipeline_detection(cart_filtered, cart_roi, best_model, 0.2, 45, 5, 2);
                    scan_vertical = found;
                } else {
                    found = pipeline_detection(cart_filtered.t(), cart_roi.t(), best_model, 0.2, 45, 5, 5);
                    scan_vertical = !found;
                }

                if (found) {
                    if(!scan_vertical) {
                        best_model.first = cv::Point2f(best_model.first.y, best_model.first.x);
                        best_model.second = cv::Point2f(best_model.second.y, best_model.second.x);
                    }

                    found = !intersectSymetricData(cart_filtered, best_model, rect_roi, rect_roi.tl());

                    if(found) {
                        cv::line(cart_out, best_model.first, best_model.second, cv::Scalar(0, 255, 0), 2, CV_AA);
                        cv::line(cart_out, cv::Point(0, cart_out.rows * 0.5), cv::Point(cart_out.cols - 1, cart_out.rows * 0.5), cv::Scalar(0, 0, 255), 1, CV_AA);
                        cv::line(cart_out, cv::Point(cart_out.cols * 0.5, 0), cv::Point(cart_out.cols * 0.5, cart_out.rows - 1), cv::Scalar(0, 0, 255), 1, CV_AA);
                    }
                }
            }

            std::cout << "Vertical: " << scan_vertical << ", Horizontal: " << !scan_vertical << std::endl;

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
