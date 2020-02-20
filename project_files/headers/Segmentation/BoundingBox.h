#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include "utils.h"

class BoundingBox
{
public:
    BoundingBox();
    ~BoundingBox();


    void get_bounding_box(cv::Mat &in_ext, cv::Mat &col_in);
};

#endif // BOUNDINGBOX_H
