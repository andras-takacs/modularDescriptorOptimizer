#include "utils.h"
#include "Mathematics/CentralMoments.h"

using namespace cv;
using namespace std;

CentralMoments::CentralMoments()
{
    cent_cy = cent_cx = inv_mu00 = mu00 = mu10 = mu01 = mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 = mu40 = mu31 = mu22 = mu13 = mu04
            = mu50 = mu41 = mu32 = mu23 = mu14 = mu05 = 0;

    nu00 = nu10 = nu01 = nu20 = nu11 = nu02 = nu30 = nu21 = nu12 = nu03 = nu40 = nu31 = nu22 = nu13 = nu04
            = nu50 = nu41 = nu32 = nu23 = nu14 = nu05 = 0;
}

CentralMoments::~CentralMoments()
{

}

CentralMoments::CentralMoments(cv::Moments& _cv_moments){

    double cx = 0, cy = 0, _inv_m00 = 0;

    if( std::abs(_cv_moments.m00) > DBL_EPSILON )
    {
        _inv_m00 = 1./_cv_moments.m00;
        cx = _cv_moments.m10*_inv_m00; cy = _cv_moments.m01*_inv_m00;
    }

    mu00 = _cv_moments.m00;
    mu10 = 0;
    mu01 = 0;
    mu20 = _cv_moments.mu02;
    mu11 = _cv_moments.mu11;
    mu02 = _cv_moments.mu02;
    mu30 = _cv_moments.mu30;
    mu21 = _cv_moments.mu21;
    mu12 = _cv_moments.mu12;
    mu03 = _cv_moments.mu03;
    inv_mu00 = 1./_cv_moments.m00;
    cent_cx = cx;
    cent_cy = cy;

    mu40 = mu31 = mu22 = mu13 = mu04 = mu50 = mu41 = mu32 = mu23 = mu14 = mu05 = 0;

    nu00 = 1;
    nu10 = 0;
    nu01 = 0;
    nu20 = _cv_moments.nu20;
    nu11 = _cv_moments.nu11;
    nu02 = _cv_moments.nu02;
    nu30 = _cv_moments.nu30;
    nu21 = _cv_moments.nu21;
    nu12 = _cv_moments.nu12;
    nu03 = _cv_moments.nu03;

    nu40 = nu31 = nu22 = nu13 = nu04 = nu50 = nu41 = nu32 = nu23 = nu14 = nu05 = 0;

}

CentralMoments::CentralMoments(RawMoments& _raw_moments){

    double cent_ms[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    CentralMoments::calculateCentralMoments(_raw_moments, cent_ms);
    std::cout<<"NU03 Norm mom: "<<cent_ms[19]<<std::endl;
    mu00 = cent_ms[0];
    mu10 = cent_ms[1];
    mu01 = cent_ms[2];
    mu20 = cent_ms[3];
    mu11 = cent_ms[4];
    mu02 = cent_ms[5];
    mu30 = cent_ms[6];
    mu21 = cent_ms[7];
    mu12 = cent_ms[8];
    mu03 = cent_ms[9];
    inv_mu00 = cent_ms[10];
    cent_cx = cent_ms[11];
    cent_cy = cent_ms[12];

    mu40 = mu31 = mu22 = mu13 = mu04 = mu50 = mu41 = mu32 = mu23 = mu14 = mu05 = 0;

    nu00 = 1;
    nu10 = 0;
    nu01 = 0;
    nu20 = cent_ms[13];
    nu11 = cent_ms[14];
    nu02 = cent_ms[15];
    nu30 = cent_ms[16];
    nu21 = cent_ms[17];
    nu12 = cent_ms[18];
    nu03 = cent_ms[19];

    nu40 = nu31 = nu22 = nu13 = nu04 = nu50 = nu41 = nu32 = nu23 = nu14 = nu05 = 0;

}

void CentralMoments::calculateCentralMoments(RawMoments& raw_moments,  double* _cent_mom){

    double cx = 0, cy = 0, _inv_m00 = 0;
    //    double ccx, ccy;
    _cent_mom[0] = raw_moments.m00;
    if( std::abs(raw_moments.m00) > DBL_EPSILON )
    {
        _inv_m00 = 1./raw_moments.m00;
        cx = raw_moments.m10*_inv_m00; cy = raw_moments.m01*_inv_m00;
    }

    _cent_mom[10] = _inv_m00;
    _cent_mom[11] = cx;
    _cent_mom[12] = cy;

    _cent_mom[3] = raw_moments.m20 - raw_moments.m10*cx;    //mu20
    _cent_mom[4] = raw_moments.m11 - raw_moments.m10*cy;    //mu11
    _cent_mom[5] = raw_moments.m02 - raw_moments.m01*cy;    //mu02

    _cent_mom[6] = raw_moments.m30 - cx*(3*_cent_mom[3] + cx*raw_moments.m10);                   //mu30
    _cent_mom[7] = raw_moments.m21 - cx*(2*_cent_mom[4] + cx*raw_moments.m01) - cy*_cent_mom[3]; //mu21
    _cent_mom[8] = raw_moments.m12 - cy*(2*_cent_mom[4] + cy*raw_moments.m10) - cx*_cent_mom[5]; //mu12
    _cent_mom[9] = raw_moments.m03 - cy*(3*_cent_mom[5] + cy*raw_moments.m01);                   //mu03

    double inv_sqrt_m00 = std::sqrt(std::abs(_inv_m00));
    double s2 = _inv_m00*_inv_m00,
           s3 = s2*inv_sqrt_m00;
    std::cout<<"Inv_m00: "<<_inv_m00<<std::endl;
    _cent_mom[13] = _cent_mom[3]*s2;     //nu20
    _cent_mom[14] = _cent_mom[4]*s2;     //nu11
    _cent_mom[15] = _cent_mom[5]*s2;     //nu02
    _cent_mom[16] = _cent_mom[6]*s3;     //nu30
    _cent_mom[17] = _cent_mom[7]*s3;     //nu21
    _cent_mom[18] = _cent_mom[8]*s3;     //nu12
    _cent_mom[19] = _cent_mom[9]*s3;     //nu03


}
