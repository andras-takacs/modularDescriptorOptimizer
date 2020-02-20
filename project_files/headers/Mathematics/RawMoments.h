#ifndef RAWMOMENTS
#define RAWMOMENTS

#include "utils.h"

using namespace cv;
using namespace std;


//!============ Raw Moments Class========================================

class RawMoments
{
public:
    //!The Default costructor;
    RawMoments();
    ~RawMoments();

    RawMoments(cv::Mat& source_im);

    //! spatial moments
    double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03, m40, m31, m22, m13, m04, m50, m41, m32, m23, m14, m05;
    //! double vector with the central moments
    //    static double raw_ms[15];
    //    cv::Mat raw_moms;

    void calculateRawMoments(cv::Mat& _source_im, double *_raw_mom);
};





#endif // RAWMOMENTS

