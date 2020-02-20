#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include "utils.h"

using namespace std;
using namespace cv;

class Images;

namespace ImageProcessing{

void loadMultipleImages(std::vector<Images> &_im_vec, uint type);

Mat illuminationInvariant(Mat &_im);

Mat gammaCorrection(Mat& _src, double gamma);

Mat getAffineTransformMat(int _image_rows, int _image_cols, const std::vector<double>& _aff_triplet);

Mat replicateBorderForEvaluation(Mat& _img, int _number_of_pixels);

KeyPoint pointTransform(KeyPoint _point, Mat _aff_transform_mat);

Mat lightIntensityChangeAndShift(Mat &_image, double _alpha, int _beta);

Mat colorTemperatureChange(Mat& _image, const std::vector<double>& _alpha, int _beta);

Images enlargeImageForRotationAndSize(Images &_src, int eval_type);

Mat normalizeWithLocalMax(Mat& _image);

Mat normalizeImage(Mat& _image);

Mat calculateEigenValues(Mat& _image);

Mat equalizeIntensity(const Mat& inputImage);

std::vector<cv::Mat> calculateGradientAngles(Mat& _inputImg, int _grad_detector);

}

#endif // IMAGEPROCESSING_H
