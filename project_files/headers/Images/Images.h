#ifndef IMAGES_H
#define IMAGES_H

#include "utils.h"

using namespace std;
using namespace cv;


class Database;

class Images
{
public:
    Images();
    Images(const cv::Size _mat_size);
    ~Images();

    Mat col_im, grey_im, mask_im;
    int rot_offset_x,rot_offset_y;

    void getImagesFromDatabase(Database &db, int position);

    void loadImages(vector<string> im_data);

    Images affineTransformImagesA(Mat &_aff_transform_mat, const std::vector<double>& _affine_triplet);
    Images affineTransformImagesRS(Mat& _aff_transform_mat);

    Images gammaCorrectImages(double gamma);




};



#endif // IMAGES_H
