#include <iostream>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <limits>

// opencv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
  * Uses the ransac algorithm to estimate lines in 2d point set.
  *
  * @param points - the 2d point set
  * @param iterations - number of iterations
  * @param threshold - max inlier distance to the model
  * @param fit_rate -  min count inliers of the point cloud in percent, so it is a valid model
  * @param best_model - best model that could be found
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

cv::Mat generateRandomImage(cv::Size size, unsigned int num_white_pixels, bool save_file = false) {
    if (size.area() < num_white_pixels)
        return cv::Mat();

    cv::Mat sample = cv::Mat::zeros(size, CV_8U);
    std::srand(std::time(0)); // use current time as seed for random generator
    for (size_t i = 0; i < num_white_pixels; i++) {
        unsigned int rand_x = std::rand() % sample.cols;
        unsigned int rand_y = std::rand() % sample.rows;
        unsigned int rand_y2 = std::rand() % (sample.rows / 50) + sample.cols * 0.5;
        sample.at<uint8_t>(rand_x, rand_y) = 255;
        sample.at<uint8_t>(rand_x, rand_y2) = 255;
    }

    if (save_file)
        cv::imwrite("sample.png", sample);

    return sample;
}

void drawStraightLine(cv::Mat& img, const std::pair<cv::Point, cv::Point>& best_model, cv::Scalar color) {
    // output
    cv::circle(img, best_model.first, 3, cv::Scalar(255,0,255), -1);
    cv::circle(img, best_model.second, 3, cv::Scalar(255,0,255), -1);

    // calculation
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

    cv::circle(img, p1, 4, cv::Scalar(255,0,255), 2);
    cv::circle(img, p2, 4, cv::Scalar(255,0,255), 2);
    cv::line(img, p, q, color, 2);
}

int main(int argc, char const *argv[]) {
    // generate a random image
    cv::Mat sample = generateRandomImage(cv::Size(600, 600), 2000);

    unsigned int num_points = cv::countNonZero(sample);

    if (num_points) {
        // 2d point list
        std::vector<cv::Point> point_list;
        cv::findNonZero(sample, point_list);

        // ransac fit line
        std::pair<cv::Point, cv::Point> best_model;
        std::vector<bool> inliers;
        double distance_average;
        fitLineRansac(point_list, 100, 100, 5, best_model, inliers, distance_average);

        // output line
        cv::Mat output;
        cv::cvtColor(sample, output, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < inliers.size(); i++)
            if(inliers[i])
                output.at<cv::Vec3b>(point_list[i]) = cv::Vec3b(0, 255, 0);
        drawStraightLine(output, best_model, cv::Scalar(0, 0, 255));
        cv::imshow("output", output);
    }

    // ransac fit line algorithm
    cv::waitKey();

    return 0;
}
