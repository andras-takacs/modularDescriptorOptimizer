#include "Mathematics/HuMomentInvariant.h"
#include "utils.h"

HuMomentInvariant::HuMomentInvariant()
{
    hu_01 = hu_02 = hu_03 = hu_04 = hu_05 = hu_06 = hu_07 = hu_08 = 0;
}

HuMomentInvariant::~HuMomentInvariant()
{

}

HuMomentInvariant::HuMomentInvariant(cv::Moments& _moms){

    double hu_ms[7] = {0,0,0,0,0,0,0};

    cv::HuMoments(_moms,hu_ms);


    double t0 = _moms.nu30 + _moms.nu12;
    double t1 = _moms.nu21 + _moms.nu03;

    double q0 = t0 * t0, q1 = t1 * t1;

//    double n4 = 4 * _moms.nu11;
//    double s = _moms.nu20 + _moms.nu02;
    double d = _moms.nu20 - _moms.nu02;

//    hu_01 = s;
//    hu_02 = d * d + n4 * _moms.nu11;
//    hu_04 = q0 + q1;
//    hu_06 = d * (q0 - q1) + n4 * t0 * t1;

    hu_01 = hu_ms[0];
    hu_02 = hu_ms[1];
    hu_03 = hu_ms[2];
    hu_04 = hu_ms[3];
    hu_05 = hu_ms[4];
    hu_06 = hu_ms[5];
    hu_07 = hu_ms[6];

//    t0 *= q0 - 3 * q1;
//    t1 *= 3 * q0 - q1;

//    double q02 = _moms.nu30 - 3 * _moms.nu12;
//    double q12 = 3 * _moms.nu21 - _moms.nu03;

//    hu_03 = q02 * q02 + q12 * q12;
//    hu_05 = q02 * t0 * (q0 - 3*q1) + q12 * t1 * (3*q0 - q1);
//    hu_07 = q12 * t0 * (q0 - 3*q1) - q02 * t1 * (3*q0 - q1);
    hu_08 = _moms.nu11 * (q0 - q1) - d * t0 * t1;


}
