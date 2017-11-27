#include <opencv2/opencv.hpp>
#include <ctime>
#include <limits>

using namespace std;
using namespace cv;

void loadImage() {
    cv::Mat src = cv::imread("cart_out.png", CV_LOAD_IMAGE_COLOR);
    cv::Rect rect(150, 180, 400, 100);
    src(rect).setTo(cv::Scalar(255,255,255));

    cv::Point center(src.cols * 0.5, src.rows - 1);

    for (size_t i = 5; i < src.cols; i += 5) {
        cv::Mat out = src.clone();
        cv::ellipse(out, center, cv::Size(i,i), 180, 0, 180, cv::Scalar(255), 1, CV_AA);

        cv::Mat out2 = cv::Mat::zeros(src.size(), CV_8U);
        cv::ellipse(out2, center, cv::Size(i,i), 180, 0, 180, cv::Scalar(255), 1, CV_AA);


        cv::imshow("out", out);
        cv::waitKey();
    }
}




int main() {
    // loadImage();

    // cv::Mat test = cv::Mat::zeros(200, 400, CV_8U);
    // int total_pixels = cv::countNonZero(test);
    // std::cout << "total_pixels: " << total_pixels << std::endl;

    cv::Mat symetric;
    std::cout << symetric.empty() << std::endl;

    return 0;
}
