#ifndef DESCRIPTOREXTRACT_H
#define DESCRIPTOREXTRACT_H

#include "utils.h"

using namespace cv;
using namespace std;

class DescriptorExtract
{
public:
    DescriptorExtract();
    DescriptorExtract(const int& _descriptorType);
    ~DescriptorExtract();

    Mat extraction(vector<KeyPoint>& _keypoints, Images& _im_in);
private:
    int descriptorType;
    Mat descriptorResults;
};

#endif // DESCRIPTOREXTRACTOR_H
