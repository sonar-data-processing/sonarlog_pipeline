#include <opencv2/opencv.hpp>
#include <ctime>
#include <limits>

using namespace std;
using namespace cv;

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


int main() {

    int iterations = 100;
    std::srand(std::time(0));

    for (size_t i = 0; i < iterations; i++) {
        cv::Mat img = cv::Mat::zeros(600,600,CV_8UC3);
        cv::Point p1(rand() % img.cols, rand() % img.rows);
        cv::Point p2(rand() % img.cols, rand() % img.rows);

        drawStraightLine(img, std::make_pair(p1, p2), cv::Scalar(0,0,255));

        imshow("Result", img);
        waitKey();
    }

    return 0;
}
