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

using namespace sonarlog_pipeline;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

using namespace cv;
using namespace std;

inline void load_sonar_holder(const base::samples::Sonar& sample, sonar_processing::SonarHolder& sonar_holder) {
    sonar_holder.Reset(sample.bins,
        rock_util::Utilities::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count);
}

// convert between two different vector types
template <class SrcType, class DstType>
void convert3(std::vector<SrcType>& src, std::vector<DstType>& dst) {
    std::copy(src.begin(), src.end(), std::back_inserter(dst));
}

// ransac fit line
bool ransac2D(std::vector<cv::Point2f>& data, std::vector<cv::Point2f>& linePoints, int minNumToFit, int iterations, float dist_threshold, bool* mask) {
    //Given:
    int good_Points             = 0;                    // the number of points fitting the model
    int trailcount              = 0;                    // count how many trails had been done
    int sizeData                = 0;                    // amount of data points
    bool* tmp_mask = NULL;

    int idx_1                   = 0;                    // index for first sample point
    int idx_2                   = 0;                    // index for second sample point
    cv::Point2f sample_point_1  = cv::Point2f(0,0);     // point 1 for model line
    cv::Point2f sample_point_2  = cv::Point2f(0,0);     // point 2 for model line
    float m                     = 0.0;                  // gradient of model
    float m_inv                 = 0.0;                  // vertical gradient to model
    float t                     = 0.0;                  // y-intercept
    float t_inv                 = 0.0;                  // y-intercept
    cv::Point2f intersection_point = cv::Point2f(0,0);  // intersection point of line trough test point and model line
    float distance_test_point   = 0;

    // to modify the required goal points based on the distance
    float d_1                   = 0.0;
    float d_2                   = 0.0;
    float d_middle              = 0.0;
    int minNumToFitModified     = 0;

    // Return:
    //vector<Point2f> linePoints(2);
    cv::Point2f goodModel1      = cv::Point2f(0,0);     // good model point 1
    cv::Point2f goodModel2      = cv::Point2f(0,0);     // good model point 2
    float m_goodModel           = 0.0;
    float t_goodModel           = 0.0;
    int goodModel_points        = 0;

    sizeData = data.size();
    if(sizeData < minNumToFit) {
        std::cout << "RANSAC ERROR: minNumToFit is to high!" << endl;
        return 0;
    }

    tmp_mask = new bool[data.size()];

  //cout << "Ransac1" << endl;
  // mask = new bool[sizeData];
    int emergency_stop = 0;           // stops while loop, when getting to many runs.
    while(trailcount < iterations)
    {
        good_Points = 0;

        // Select random subset -> hypothetical inliers
        idx_1 = rand() % sizeData;
        while((data[idx_1].x == 0) && (data[idx_1].y == 0) && emergency_stop < 100) {
            idx_1 = rand() % sizeData;
            emergency_stop++;
        }

        emergency_stop = 0;
        idx_2 = rand() % sizeData;

        while(((idx_1 == idx_2) or ((data[idx_2].x == 0) && (data[idx_2].y == 0))) && emergency_stop < 100) {
            idx_2 = rand() % sizeData;
            emergency_stop++;
        }

        sample_point_1.x = data[idx_1].x;
        sample_point_1.y = data[idx_1].y;
        sample_point_2.x = data[idx_2].x;
        sample_point_2.y = data[idx_2].y;
    //
        // check if one of the sample points is at 0/0

        // fit model to hypothetical inlier
        // line for model: y = mx + t
        //model.x = sample_point_2.x - sample_point_1.x;
        //model.y = sample_point_2.y - sample_point_1.y;
        // m = (y2-y1)/(x2-x1)
        m = (sample_point_2.y - sample_point_1.y) / (sample_point_2.x - sample_point_1.x);
        t = sample_point_2.y - m * sample_point_1.x;

        // test all other data -> consensus set
        // line for test point: y_testpoint = m_inv * x_testpoint + t_inv
        // m` = m_inv = -1/m =  (x2-x1) / (y2-y1) => m is vertical to m_inv
        m_inv = -1 / m;
        for(int i = 0; i < sizeData; i++) {
            // t_inv = data[i].y - m_inv * data[i].x = data[i].y + data[i].x / m;
            t_inv = data[i].y - m_inv * data[i].x;
            // intersection point of model and test line
            intersection_point.x = (t_inv - t) / (m - m_inv);
            intersection_point.y = m * intersection_point.x +t;

            // distance between intersection point and test point
            // vector d = data - intersection point
            // distance = |d| = sqrt(d.x² + d.y²)
            distance_test_point = sqrt( pow((data[i].x - intersection_point.x),2.0) + pow((data[i].y - intersection_point.y),2.0));
            if((distance_test_point <= dist_threshold) && ((data[i].x > 0.0) || (data[i].y > 0.0))) {
                tmp_mask[i] = true;
                good_Points++;
            }
            else {
              tmp_mask[i] = false;
            }
        }
        // check if model is reasonably

        /*
         * change minNumToFit based on the distance between object and scanner
         */
        // calculate middle distance of mirror points to scanner
        d_1 = sqrt( pow((sample_point_1.x),2.0) + pow((sample_point_1.y),2.0));
        d_2 = sqrt( pow((sample_point_2.x),2.0) + pow((sample_point_2.y),2.0));
        d_middle = (d_1 + d_2) / 2;

          minNumToFitModified = minNumToFit;

          int test = 0;
          if((good_Points > goodModel_points) && (good_Points >= minNumToFitModified))
          {
            goodModel1.x = sample_point_1.x;
            goodModel1.y = sample_point_1.y;
            goodModel2.x = sample_point_2.x;
            goodModel2.y = sample_point_2.y;
            m_goodModel = m;
            t_goodModel = t;
            goodModel_points = good_Points;
            for(int i= 0; i < data.size(); i++)
            {
              mask[i] = tmp_mask[i];
              test ++;
            }
          }
        test = 0;

          // improve model
          trailcount++;
        }

        if((goodModel1.x != 0.0) && (goodModel1.y != 0.0) && (goodModel2.x != 0.0) && (goodModel2.y != 0.0))
        {
            linePoints.push_back(cv::Point2f(goodModel1.x, goodModel1.y));
            linePoints.push_back(cv::Point2f(goodModel2.x, goodModel2.y));
        // linePoints[0].x = goodModel1.x;
        // linePoints[0].y = goodModel1.y;
        // linePoints[1].x = goodModel2.x;
        // linePoints[1].y = goodModel2.y;

        delete tmp_mask;
        return 1;
        }
        delete tmp_mask;
        return 0;
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/pipeline-front.0.log",
        DATA_PATH_STRING + "/logs/pipeline-front.1.log",
        DATA_PATH_STRING + "/logs/pipeline-parallel.0.log"
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
            preprocessing::adaptative_clahe(cart_image, cart_aux);
            cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(5, 5));
            cv::morphologyEx(cart_mask, cart_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(9, 9)), cv::Point(-1, -1), 2);
            cart_aux.copyTo(cart_filtered, cart_mask);

            /* segmentation */
            cv::Mat cart_thresh, cart_aux2;
            cart_aux2 = cart_filtered < 50;
            cart_aux2.copyTo(cart_thresh, cart_mask);

            /* 2d point list */
            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(cart_thresh.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
            std::vector<cv::Point> point_list_2i;
            for (size_t i = 0; i < contours.size(); i++)
                for (size_t j = 0; j < contours[i].size(); j++)
                    point_list_2i.push_back(contours[i][j]);

            std::vector<cv::Point2f> point_list;
            convert3(point_list_2i, point_list);

            std::vector<cv::Point2f> line_points;
            int minNumToFit = 100;
            int iterations = 40;
            float dist_threshold = 2;
            std::vector<bool*> mask_line_points(1);
            mask_line_points[0] = new bool[point_list.size()];

            bool foundLine = ransac2D(point_list, line_points, minNumToFit, iterations, dist_threshold, mask_line_points[0]);
            std::cout << "FoundLine " << foundLine << std::endl;
            if (foundLine) {
                cv::Mat cart_out;
                cv::cvtColor(cart_image, cart_out, cv::COLOR_GRAY2BGR);
                cv::line(cart_out, line_points[0], line_points[1], cv::Scalar(255,0,0), 2, CV_AA);
                cv::imshow("cart_out", cart_out);
            }





            /* output */
            cv::imshow("cart_raw", cart_raw);
            cv::imshow("cart_denoised", cart_denoised);
            cv::imshow("cart_image", cart_image);
            cv::imshow("cart_filtered", cart_filtered);
            cv::imshow("cart_thresh", cart_thresh);
            // cv::imshow("cart_out", cart_out);
            cv::waitKey(0);
        }
        cv::waitKey(0);
    }

return 0;
}
