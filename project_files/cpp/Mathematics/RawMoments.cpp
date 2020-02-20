
#include "utils.h"
#include "Mathematics/RawMoments.h"

using namespace cv;
using namespace std;


RawMoments::RawMoments(){

    m00 = m10 = m01 = m20 = m11 = m02 = m30 = m21 = m12 = m03 = m40 = m31 = m22 = m13 = m04
            = m50 = m41 = m32 = m23 = m14 = m05= 0;
}

RawMoments::~RawMoments(){

}

RawMoments::RawMoments(cv::Mat& _source_im){

    double raw_ms[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    RawMoments::calculateRawMoments(_source_im, raw_ms);

    m00 = raw_ms[0];
    m10 = raw_ms[1];
    m01 = raw_ms[2];
    m20 = raw_ms[3];
    m11 = raw_ms[4];
    m02 = raw_ms[5];
    m30 = raw_ms[6];
    m21 = raw_ms[7];
    m12 = raw_ms[8];
    m03 = raw_ms[9];
    m40 = raw_ms[10];
    m31 = raw_ms[11];
    m22 = raw_ms[12];
    m13 = raw_ms[13];
    m04 = raw_ms[14];
    m50 = raw_ms[15];
    m41 = raw_ms[16];
    m32 = raw_ms[17];
    m23 = raw_ms[18];
    m14 = raw_ms[19];
    m05 = raw_ms[20];


}
//!Code is taken from OpenCV version 3.0.0 imgproc section moments.cpp
//!Added part is to calculate the fourth degree raw moment elementst
void RawMoments::calculateRawMoments(cv::Mat& _source_im, double* _raw_mom){

    cv::Size size = _source_im.size();
    int x, y;
    double mom[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for( y = 0; y < size.height; ++y)
    {
//        const uchar* ptr = (const uchar*)(_source_im.data + y*_source_im.step);
        double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;


        for( x = 0; x < size.width; ++x )
        {
//                        uchar p = ptr[x], xp = x * p, xxp, xxxp, xxxxp;
            double p = _source_im.at<uchar>(y,x), xp = x * p, xxp, xxxp, xxxxp;


            x0 += p;
            x1 += xp;
            xxp = xp * x;
            x2 += xxp;
            xxxp = xxp * x;
            x3 += xxxp;
            xxxxp = xxxp * x;
            x4 += xxxxp;
            x5 += xxxxp * x;

            //            if(x==125)
            //            std::cout<<"ptr: "<<p<<std::endl;

        }

        double py = y * x0, sy = y*y, ssy = sy*y, sssy = ssy * y;

        mom[20] += py * sssy;      // m05
        mom[19] += x1 * sssy;      // m14
        mom[18] += x2 * ssy;       // m23
        mom[17] += x3 * sy;        // m32
        mom[16] += x4 * y;         // m41
        mom[15] += x5;             // m50
        mom[14] += py * ssy;       // m04
        mom[13] += x1 * ssy;       // m13
        mom[12] += x2 * sy;        // m22
        mom[11] += x3 * y;         // m31
        mom[10] += x4;             // m40
        mom[9]  += py * sy;        // m03
        mom[8]  += x1 * sy;        // m12
        mom[7]  += x2 * y;         // m21
        mom[6]  += x3;             // m30
        mom[5]  += x0 * sy;        // m02
        mom[4]  += x1 * y;         // m11
        mom[3]  += x2;             // m20
        mom[2]  += py;             // m01
        mom[1]  += x1;             // m10
        mom[0]  += x0;             // m00
    }


    for( x = 0; x < 21; ++x )
        _raw_mom[x] = (double)mom[x];
}
