#ifndef SEGMENTATIONEVALUATION_H
#define SEGMENTATIONEVALUATION_H

#include "utils.h"
#include "Evaluation/EvaluationValues.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


class SegmentationEvaluation
{
public:
    SegmentationEvaluation();
    ~SegmentationEvaluation();

//! SETUP
bool mark_keypoints;
int matcher_type, evaluation_type, descriptor_type, comparation_method;
int maxGamma, minGamma, maxGaussian;
Point2f destinationTriplet;

//! RESULTS
Mat img_matches, altered_im;
vector<QVector<double> >plot_vector, x_average_vector, y_average_vector;
QVector<double> x_vector, y_vector;
double descriptor_count_time;



//void setVariables(Images _eval_images, uint _eval_type, bool _mark_keypoints, int _rotation_angle, int _gaussian_blur);

void setVariables(int _desc_type, int _matcher_type, int _evaluation_type, int _comparation_method);

void setIlluminationVariables(int _desc_type, int _matcher_type, int _max_gamma, int _min_gamma);

void markKeyPoints();

void evaluation(vector<Images> &_eval_images);

DMatch euclidWithKnnSearch(Mat& baseVector, int baseVectorAt, Mat& compareMat);




};

#endif // SEGMENTATIONEVALUATION_H
