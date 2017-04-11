#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    cv::Mat img = cv::Mat::zeros(500,500,CV_8U);
    img.rowRange(100, 400) = 255;

    cv::Point p1(250,50);
    cv::Point p2(250,200);
    cv::Point p3(250,350);
    cv::Point p4(250,450);

    cv::Mat out;
    cv::cvtColor(img, out, CV_GRAY2BGR);
    cv::circle(out, p1, 1, cv::Scalar(0,0,255), -1);
    cv::circle(out, p2, 1, cv::Scalar(0,0,255), -1);
    cv::circle(out, p3, 1, cv::Scalar(0,0,255), -1);
    cv::circle(out, p4, 1, cv::Scalar(0,0,255), -1);

    std::cout << "Contains P1? " << (img.at<uchar>(p1) == 255) << std::endl;
    std::cout << "Contains P2? " << (img.at<uchar>(p2) == 255) << std::endl;
    std::cout << "Contains P3? " << (img.at<uchar>(p3) == 255) << std::endl;
    std::cout << "Contains P4? " << (img.at<uchar>(p4) == 255) << std::endl;

    imshow("Result", img);
    imshow("Result2", out);
    waitKey();

    return 0;
}
