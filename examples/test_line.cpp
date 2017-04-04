#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void getLinePointinImageBorder(const cv::Point& p1_in, const cv::Point& p2_in,
                               cv::Point& p1_out, cv::Point& p2_out,
                               int rows, int cols)
{
    double m = (double) (p1_in.y - p2_in.y) / (double) (p1_in.x - p2_in.x + std::numeric_limits<double>::epsilon());
    double b = p1_in.y - (m * p1_in.x);

    std::vector<cv::Point> border_point;
    double x,y;
    //test for the line y = 0
    y = 0;
    x = (y-b)/m;
    if(x >= 0 && x <= cols)
        border_point.push_back(cv::Point(x,y));

    //test for the line y = img.rows
    y = rows;
    x = (y-b)/m;
    if(x >= 0 && x <= cols)
        border_point.push_back(cv::Point(x,y));

    //check intersection with horizontal lines x = 0
    x = 0;
    y = m * x + b;
    if(y >= 0 && y <= rows)
        border_point.push_back(cv::Point(x,y));

    x = cols;
    y = m * x + b;
    if(y >= 0 && y <= rows)
        border_point.push_back(cv::Point(x,y));

    p1_out = border_point[0];
    p2_out = border_point[1];
}

int main() {

    cv::Mat img = cv::Mat::zeros(600,600,CV_8UC3);
    cv::Point p1(307,586);
    cv::Point p2(307,588);

    std::cout << "Reference points:" << p1 << "," << p2 << std::endl;

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

    std::cout << "Final points: " << p << "," << q << std::endl;
    getLinePointinImageBorder(p,q,p,q,img.rows,img.cols);
    // cv::clipLine(img.size(), p, q);
    std::cout << "Final points(CL): " << p << "," << q << std::endl;

    cv::circle(img, p1, 4, cv::Scalar(255,0,255), -1);
    cv::circle(img, p2, 4, cv::Scalar(255,0,255), -1);
    // cv::Point p, q;
    // getLinePointinImageBorder(p1, p2, p, q, img.rows, img.cols);
    cv::line(img, p, q, cv::Scalar(0,0,255), 2);

    imshow("Result", img);
    waitKey();

    return 0;
}
