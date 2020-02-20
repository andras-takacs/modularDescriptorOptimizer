#ifndef KEYINTERESTPOINT_H
#define KEYINTERESTPOINT_H

#include "utils.h"


using namespace std;
using namespace cv;


class KeyInterestPoint
{
public:
    KeyInterestPoint();
    ~KeyInterestPoint();


    //    std::vector<ImageGroup> &bum;

    void extractDescriptorsAndLables(const std::string& testOrTrain,
                                     const std::string& keyPointType,
                                     const int &descriptorType,
                                     cv::Mat &descrip,
                                     cv::Mat &label,
                                     std::vector<Images> &im,
                                     int threshold,
                                     int loop_start,
                                     int loop_finish,
                                     const int _radius);

    void extraction(const int& testOrTrain,
                    const int& keyPointType,
                    const int& labelType,
                    const int& descriptorType,
                    cv::Mat &descrip,
                    cv::Mat &label,
                    const int& train_percent,
                    const int _radius);

    std::vector<cv::KeyPoint> keypoints;

};

#endif // KEYINTERESTPOINT_H
