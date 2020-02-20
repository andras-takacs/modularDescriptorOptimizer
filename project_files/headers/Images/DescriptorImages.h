#ifndef DESCRIPTORIMAGES_H
#define DESCRIPTORIMAGES_H

#include "utils.h"

using namespace cv;
using namespace std;

class MDADescriptor;

class DescriptorImages
{
public:
    DescriptorImages();
    ~DescriptorImages();

     Mat blured_rgb_img,blured_hsv_img, blured_gray_img, dist_trans_img;
     Mat converted_img, float_img, gevers3l, gevers3c, geusebroekHC, normRGB;
     std::vector<cv::Mat>gradient_angles;
    std::shared_ptr<MDADescriptor> currDescriptor;


     void prepareImageForDescriptor(Mat const& _img);
     void prepareImageForTestDescriptor(const Mat &_img);
     void prepareImageForModularDescriptor(const Mat &_img,std::shared_ptr<MDADescriptor> _descriptor);
     Mat genomeSpecifiedSpaceConversion(int _genome, Mat& _img);
     Mat normalizeImage(Mat& _img);
};

#endif // DESCRIPTORIMAGES_H
