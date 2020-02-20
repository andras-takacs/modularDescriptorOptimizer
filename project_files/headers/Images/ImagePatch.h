#ifndef IMAGEPATCH_H
#define IMAGEPATCH_H

#include "utils.h"

using namespace cv;
using namespace std;

class ImagePatch
{
public:
    ImagePatch();
    ~ImagePatch();
    ImagePatch(int _width, int _height, int _radius);

    int size(){return patch_size;}
    int width(){return patch_width;}
    int height(){return patch_height;}
    int img_height(){return full_img_height;}
    int img_width(){return full_img_width;}
    int radius(){return patch_radius;}

    vector<Mat> hsv_channels, rgb_channels, converted_channels, float_im_channels, angle_channels;
    Mat src_image, rgb_image, hsv_image, distance_transfer;
    Mat gevers_3l_mat,gevers_3c_mat,geusebroek_HC_mat,converted_imge;
    cv::KeyPoint centerPoint;

    void setFullImageWidth(int _width){full_img_width=_width;}
    void setFullImageHeight(int _height){full_img_height=_height;}
    void setRadius(int _radius){patch_radius = _radius;}

private:
    int patch_size, patch_width, patch_height, patch_radius;
    int full_img_width, full_img_height;

};

#endif // IMAGEPATCH_H
