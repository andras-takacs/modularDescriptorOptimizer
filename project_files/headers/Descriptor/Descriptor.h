#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "utils.h"

using namespace cv;
using namespace std;

class Descriptor
{
public:
    Descriptor();
    ~Descriptor();

    Mat descriptors;

    void descriptor_feature (vector<KeyPoint> &keypoint,
                             Mat const&image,
                             const int &descriptorType, const int _radius=4);

    void rgbDescriptor(Mat const& _img, vector<KeyPoint> const&keypoint, uint _patch_size, bool seeSize);

    void projectDescriptor(Mat const& _img, vector<KeyPoint> const&keypoint, uint _patch_size, bool seeSize);

    void doctorateDescriptor(Mat const& _img, vector<KeyPoint> const&keypoint, uint _patch_size, bool seeSize);

    void isFloatMatrix(Mat &rawMat);

};

#endif // DESCRIPTOR_H
