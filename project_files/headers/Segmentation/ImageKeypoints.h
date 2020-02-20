#ifndef IMAGEKEYPOINTS_H
#define IMAGEKEYPOINTS_H

#include "utils.h"

using namespace cv;
using namespace std;

class ImageKeypoints
{
public:
    ImageKeypoints();
    ImageKeypoints(const int& keyPointType, Mat& _sample_image, const int &_image_margin, int _radius = 4);
    ~ImageKeypoints();

    std::vector<cv::KeyPoint> keypoints, transformedKeypoints, homographyKeypoints;

    void calculateCoordinates(Images &im_in);
    bool theImageIsDifferent(Mat& _image_in);
    cv::KeyPoint pointTransform(KeyPoint _point, Mat _aff_transform_mat);
    void transformAllKeypoints(Mat _aff_transform_mat);

    vector<KeyPoint> getKeyPoints(){return keypoints;}
    int getImageMargin(){return imageMargin;}
    void setImageMargin(int _image_margin){imageMargin = _image_margin;}

    void homographyTransformAllKeypoints(Mat& _homographyMat);
    bool pointIsOutOfRange(cv::Point2f _point);

    void kpType(int _kpType){keyPointType=_kpType;}
    int kpType(){return keyPointType;}

private:

    int keyPointType, keyPointRadius, imageWidth, imageHeight, imageMargin;
};

#endif // IMAGEKEYPOINTS_H
