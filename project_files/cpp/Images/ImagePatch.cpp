#include "Images/ImagePatch.h"

ImagePatch::ImagePatch()
{
    src_image = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    rgb_image = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    hsv_image = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    distance_transfer = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
//    edges_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
//    dist_trans_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
//    eigen_vals = cv::Mat(1,1,CV_32FC(6),cv::Scalar::all(0));
}

ImagePatch::~ImagePatch()
{    

}

ImagePatch::ImagePatch(int _width, int _height, int _radius)
{
patch_width = _width;
patch_height = _height;
patch_radius = _radius;

src_image = cv::Mat(_height,_width,CV_8UC3,cv::Scalar::all(0));
rgb_image = cv::Mat(_height,_width,CV_8UC3,cv::Scalar::all(0));
hsv_image = cv::Mat(_height,_width,CV_8UC3,cv::Scalar::all(0));
distance_transfer = cv::Mat(_height,_width,CV_8UC3,cv::Scalar::all(0));

}
