#ifndef MATEK_H
#define MATEK_H

#include "utils.h"

using namespace std;
using namespace cv;

namespace Matek
{
int decimalToBinary(int dec);
int binaryToDecimal(int bin);
int quickPow10(int pow);
int binVectorToInt(std::vector<int> _input);

Mat elemMatrixTransform(cv::Mat& _inputMat, cv::Mat& _transformationMat);
Mat colorNormalization(cv::Mat& _inputMat);
Mat opponentColorMat(cv::Mat& _input_rgb_image);
Mat invariantLoneLtwoLthree(Mat& _input_rgb_image);
Mat invariantConeCtwoCthree(Mat& _input_rgb_image);
Mat featureHC(Mat& _float_input_rgb_im);
std::vector<float> calculateAngleDifferences(std::vector<float>_incoming_angles);

}

#endif // MATEK_H
