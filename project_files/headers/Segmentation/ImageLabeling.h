#ifndef IMAGELABELING_H
#define IMAGELABELING_H

#include "utils.h"

using namespace cv;
using namespace std;

class ImageLabeling
{
public:
    ImageLabeling();
    ImageLabeling(const int& _labelType);
    ~ImageLabeling();

    Mat labelingImage(Images &_im_in, vector<KeyPoint> &_keypoints);

private:
    int labelType;
};

#endif // IMAGELABELING_H
