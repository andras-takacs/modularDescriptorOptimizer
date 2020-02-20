#include "Mathematics/AffineMoments.h"
#include "utils.h"

using namespace cv;
using namespace std;

AffineMoments::AffineMoments()
{
    ami_01 = ami_02 = ami_03 = ami_04 = ami_05 = ami_06 = ami_07 = ami_08 = ami_09 = ami_10 = 0;
}


AffineMoments::~AffineMoments()
{

}

AffineMoments::AffineMoments(CentralMoments& _cent_moments){

    double amis[10] = {0,0,0,0,0,0,0,0,0,0};

    AffineMoments::calculateAllInvariants(_cent_moments, amis);

    ami_01 = amis[0];
    ami_02 = amis[1];
    ami_03 = amis[2];
    ami_04 = amis[3];
    ami_05 = amis[4];
    ami_06 = amis[5];
    ami_07 = amis[6];
    ami_08 = amis[7];
    ami_09 = amis[8];
    ami_10 = amis[9];

}


double AffineMoments::calculateSingleInvariant(CentralMoments& _moms, int numeration_of_invariant){

    double invariant = 0;

    double  sq20, sq11, sq02, sq30, sq21, sq12, sq03, sq31, sq22, sq13, sq41, sq32, sq23, sq14;
    double  cc11, cc12, cc21;
    double  mm00_2, mm00_4, mm00_6, mm00_8, mm00_7;


    //    double  sq20 = _moms.nu20*_moms.nu20,
    //            sq11 = _moms.nu11*_moms.nu11,
    //            sq02 = _moms.nu02*_moms.nu02,
    //            sq30 = _moms.nu30*_moms.nu30,
    //            sq21 = _moms.nu21*_moms.nu21,
    //            sq12 = _moms.nu12*_moms.nu12,
    //            sq03 = _moms.nu03*_moms.nu03,
    //            sq31 = _moms.nu31*_moms.nu31,
    //            sq22 = _moms.nu22*_moms.nu22,
    //            sq13 = _moms.nu13*_moms.nu13,
    //            sq41 = _moms.nu41*_moms.nu41,
    //            sq32 = _moms.nu32*_moms.nu32,
    //            sq23 = _moms.nu23*_moms.nu23,
    //            sq14 = _moms.nu14*_moms.nu14;

    //    double  cc11 = sq11*_moms.nu11,
    //            cc12 = sq12*_moms.nu12,
    //            cc21 = sq21*_moms.nu21;

    mm00_2 = _moms.nu00*_moms.nu00;
    mm00_4 = mm00_2*mm00_2;


    switch (numeration_of_invariant) {
    case 0:
        std::cout << "The numbering of Affine Moment Invariants start at 1 ! Choose a number between 1 and 10!" << std::endl;
        break;

    case 1:
        sq11 = _moms.nu11 * _moms.nu11;

        invariant = (_moms.nu20 * _moms.nu02 - sq11) / mm00_4;

        break;
    case 2:
        sq30 = _moms.nu30*_moms.nu30;
        sq03 = _moms.nu03*_moms.nu03;
        sq12 = _moms.nu12*_moms.nu12;

        mm00_8 = mm00_4*mm00_4;

        invariant = (-sq30 * sq03 + 6 * _moms.nu30 * _moms.nu21 * _moms.nu12 * _moms.nu03 - 4 * _moms.nu30 * sq12 * _moms.nu12) / (mm00_8*mm00_2);

        break;
    case 3:
        sq12 = _moms.nu12*_moms.nu12;
        sq21 = _moms.nu21*_moms.nu21;

        mm00_6 = mm00_4*mm00_2;

        invariant = (_moms.nu20 * _moms.nu21 * _moms.nu03 - _moms.nu20 * sq12 - _moms.nu11 * _moms.nu30 * _moms.nu03
                + _moms.nu11 * _moms.nu21 * _moms.nu12 + _moms.nu02 * _moms.nu30 * _moms.nu12 - _moms.nu02 * sq21) / (mm00_6*_moms.nu00);
        break;
    case 4:
        sq20 = _moms.nu20*_moms.nu20;
        sq11 = _moms.nu11*_moms.nu11;
        sq02 = _moms.nu02*_moms.nu02;
        sq30 = _moms.nu30*_moms.nu30;
        sq21 = _moms.nu21*_moms.nu21;
        sq12 = _moms.nu12*_moms.nu12;
        sq03 = _moms.nu03*_moms.nu03;

        cc11 = sq11*_moms.nu11;

        mm00_6 = mm00_4*mm00_2;
        mm00_7 = mm00_6*_moms.nu00;

        invariant = (-sq20*_moms.nu20 * sq03 + 6 * sq20 * _moms.nu11 * _moms.nu12 * _moms.nu03 - 3 * sq20 * _moms.nu02 *sq12
                - 6 * _moms.nu20 * sq11 * _moms.nu21 * _moms.nu03 - 6 * _moms.nu20 * sq11 * sq12
                + 12 * _moms.nu20 * _moms.nu11 * _moms.nu02 * _moms.nu21 * _moms.nu12 - 3 * _moms.nu20 * sq02 * sq21
                + 2 * cc11 * _moms.nu30 * _moms.nu03 + 6 * cc11 * _moms.nu21 * _moms.nu12 - 6 * sq11 * _moms.nu02 * _moms.nu30 * _moms.nu12
                - 6 * sq11 * _moms.nu02 * sq21 + 6 * _moms.nu11 * sq02 * _moms.nu30 * _moms.nu21
                - 1 * sq02 * _moms.nu02 * sq30)/ (mm00_7 * mm00_4);
        break;
    case 5:
        sq22 = _moms.nu22*_moms.nu22;

        invariant = (_moms.nu40 * _moms.nu04 - 4 * _moms.nu31 * _moms.nu13 + 3 * sq22)/ (mm00_4*mm00_2);
        break;
    case 6:
        sq31 = _moms.nu31*_moms.nu31;
        sq22 = _moms.nu22*_moms.nu22;
        sq13 = _moms.nu13*_moms.nu13;

        mm00_6 = mm00_4*mm00_2;
        mm00_7 = mm00_6*_moms.nu00;

        invariant = (_moms.nu40 * _moms.nu22 * _moms.nu04 - _moms.nu40 * sq13 - sq31 * _moms.nu04
                + 2 * _moms.nu31 * _moms.nu22 * _moms.nu13- sq22) / (mm00_7 * mm00_2);

        break;
    case 7:
        sq20 = _moms.nu20*_moms.nu20;
        sq11 = _moms.nu11*_moms.nu11;
        sq02 = _moms.nu02*_moms.nu02;

        mm00_6 = mm00_4*mm00_2;

        invariant = (sq20 * _moms.nu04 - 4 * _moms.nu20 * _moms.nu11 * _moms.nu13 + 2 * _moms.nu20 * _moms.nu02 * _moms.nu22 + 4 * sq11 * _moms.nu22
                - 4 * _moms.nu11 * _moms.nu02 * _moms.nu31 + sq02 * _moms.nu40) / (mm00_6*_moms.nu00);
        break;
    case 8:
        sq20 = _moms.nu20*_moms.nu20;
        sq11 = _moms.nu11*_moms.nu11;
        sq02 = _moms.nu02*_moms.nu02;
        sq13 = _moms.nu13*_moms.nu13;
        sq22 = _moms.nu22*_moms.nu22;
        sq31 = _moms.nu31*_moms.nu31;

        mm00_8 = mm00_4*mm00_4;

        invariant = (sq20 * _moms.nu22 * _moms.nu04 - sq20*sq13 - 2 * _moms.nu20 * _moms.nu11 * _moms.nu31 * _moms.nu04
                + 2 * _moms.nu20 * _moms.nu11 * _moms.nu22 * _moms.nu13 + _moms.nu20 * _moms.nu02 * _moms.nu40 * _moms.nu04
                - 2 * _moms.nu20 * _moms.nu02 * _moms.nu31 * _moms.nu13 + _moms.nu20 * _moms.nu02 * sq22 + 4 * sq11 * _moms.nu31 * _moms.nu13
                - 4 * sq11 * sq22 - 2 * _moms.nu11 * _moms.nu02 * _moms.nu40 * _moms.nu13 + 2 * _moms.nu11 * _moms.nu02 * _moms.nu31 * _moms.nu22
                + sq02 * _moms.nu40 * _moms.nu22 - sq02 * sq31) / (mm00_8*mm00_2);
        break;
    case 9:
        sq30 = _moms.nu30*_moms.nu30;
        sq21 = _moms.nu21*_moms.nu21;
        sq12 = _moms.nu12*_moms.nu12;
        sq03 = _moms.nu03*_moms.nu03;

        cc12 = sq12*_moms.nu12;
        cc21 = sq21*_moms.nu21;

        mm00_6 = mm00_4*mm00_2;
        mm00_7 = mm00_6*_moms.nu00;

        invariant = (sq30 * sq12 * _moms.nu04 - 2 * sq30 * _moms.nu12 * _moms.nu03 * _moms.nu13 + sq30 * sq03 * _moms.nu22
                - 2 * _moms.nu30 * sq21 * _moms.nu12 * _moms.nu04 + 2 * _moms.nu30 * sq21 * _moms.nu03 * _moms.nu13
                + 2 * _moms.nu30 * _moms.nu21 * sq12 * _moms.nu13 - 2 * _moms.nu30 * _moms.nu21 * sq03 * _moms.nu31
                - 2 * _moms.nu30 * cc12 * _moms.nu22 + 2 * _moms.nu30 * sq12 * _moms.nu03 * _moms.nu31 + 4 * sq21 * sq21 * _moms.nu04
                - 2 * cc21 * _moms.nu12 * _moms.nu13 - 2 * cc21 * _moms.nu03 * _moms.nu22 + 3 * sq21 * sq12 * _moms.nu22
                + 2 * sq21 * _moms.nu12 * _moms.nu03 * _moms.nu31 + sq21 * sq03 * _moms.nu40 - 2 * _moms.nu21 * cc12 * _moms.nu31
                - 2 * _moms.nu21 * sq12 * _moms.nu03 * _moms.nu40 + sq12 * sq12 * _moms.nu40) / (mm00_7 * mm00_6);
        break;
    case 10:
        sq41 = _moms.nu41*_moms.nu41;
        sq32 = _moms.nu32*_moms.nu32;
        sq23 = _moms.nu23*_moms.nu23;
        sq14 = _moms.nu14*_moms.nu14;

        mm00_6 = mm00_4*mm00_2;
        mm00_7 = mm00_6*_moms.nu00;

        invariant = (-_moms.nu50 * _moms.nu50 * _moms.nu05 * _moms.nu05 + 10 * _moms.nu50 * _moms.nu41 * _moms.nu14 * _moms.nu05
                - 4 * _moms.nu50 * _moms.nu32 * _moms.nu23 * _moms.nu05 - 16 * _moms.nu50 * _moms.nu32 * sq14 + 12 * _moms.nu50 * sq23 * _moms.nu14
                - 16 * sq41 * _moms.nu23 * _moms.nu05 - 9 * sq41 * sq14 + 12 * _moms.nu41 * sq32 * _moms.nu05
                + 76 * _moms.nu41 * _moms.nu32 * _moms.nu23 * _moms.nu14 - 48 * _moms.nu41 * _moms.nu23 * sq23
                - 48 * _moms.nu32 * sq32 * _moms.nu14 + 32 * sq32 * sq23) / (mm00_7 * mm00_7);
        break;

    default:
        std::cout << "There is no more than 10 Affine Moment Invariants! Choose a number between 1 and 10!" << std::endl;
        break;
    }

    return invariant;

}

void AffineMoments::calculateAllInvariants(CentralMoments& _moms, double* _amis){


    double  sq20 = _moms.nu20*_moms.nu20,
            sq11 = _moms.nu11*_moms.nu11,
            sq02 = _moms.nu02*_moms.nu02,
            sq30 = _moms.nu30*_moms.nu30,
            sq21 = _moms.nu21*_moms.nu21,
            sq12 = _moms.nu12*_moms.nu12,
            sq03 = _moms.nu03*_moms.nu03,
            sq31 = _moms.nu31*_moms.nu31,
            sq22 = _moms.nu22*_moms.nu22,
            sq13 = _moms.nu13*_moms.nu13,
            sq41 = _moms.nu41*_moms.nu41,
            sq32 = _moms.nu32*_moms.nu32,
            sq23 = _moms.nu23*_moms.nu23,
            sq14 = _moms.nu14*_moms.nu14;

    double  cc11 = sq11*_moms.nu11,
            cc12 = sq12*_moms.nu12,
            cc21 = sq21*_moms.nu21;

    double  inv_nu00 = 1./_moms.nu00,
            mm00_2 = _moms.nu00*_moms.nu00,
            mm00_4 = mm00_2*mm00_2,
            mm00_6 = mm00_4*mm00_2,
            mm00_8 = mm00_4*mm00_4,
            mm00_7 = mm00_8*inv_nu00,
            mm00_10 = mm00_8*mm00_2;


    _amis[0] = (_moms.nu20 * _moms.nu02 - sq11) / mm00_4;

    _amis[1] = (-sq30 * sq03 + 6 * _moms.nu30 * _moms.nu21 * _moms.nu12 * _moms.nu03 - 4 * _moms.nu30 * sq12 * _moms.nu12) / mm00_10;

    _amis[2] = (_moms.nu20 * _moms.nu21 * _moms.nu03 - _moms.nu20 * sq12 - _moms.nu11 * _moms.nu30 * _moms.nu03
                + _moms.nu11 * _moms.nu21 * _moms.nu12 + _moms.nu02 * _moms.nu30 * _moms.nu12 - _moms.nu02 * sq21) / mm00_7;

    _amis[3] = (-sq20*_moms.nu20 * sq03 + 6 * sq20 * _moms.nu11 * _moms.nu12 * _moms.nu03 - 3 * sq20 * _moms.nu02 *sq12
                - 6 * _moms.nu20 * sq11 * _moms.nu21 * _moms.nu03 - 6 * _moms.nu20 * sq11 * sq12
                + 12 * _moms.nu20 * _moms.nu11 * _moms.nu02 * _moms.nu21 * _moms.nu12 - 3 * _moms.nu20 * sq02 * sq21
                + 2 * cc11 * _moms.nu30 * _moms.nu03 + 6 * cc11 * _moms.nu21 * _moms.nu12 - 6 * sq11 * _moms.nu02 * _moms.nu30 * _moms.nu12
                - 6 * sq11 * _moms.nu02 * sq21 + 6 * _moms.nu11 * sq02 * _moms.nu30 * _moms.nu21
                - 1 * sq02 * _moms.nu02 * sq30)/ (mm00_7 * mm00_4);

    _amis[4] = (_moms.nu40 * _moms.nu04 - 4 * _moms.nu31 * _moms.nu13 + 3 * sq22)/ mm00_6;

    _amis[5] = (_moms.nu40 * _moms.nu22 * _moms.nu04 - _moms.nu40 * sq13 - sq31 * _moms.nu04
                + 2 * _moms.nu31 * _moms.nu22 * _moms.nu13- sq22) / (mm00_7 * mm00_2);

    _amis[6] = (sq20 * _moms.nu04 - 4 * _moms.nu20 * _moms.nu11 * _moms.nu13 + 2 * _moms.nu20 * _moms.nu02 * _moms.nu22 + 4 * sq11 * _moms.nu22
                - 4 * _moms.nu11 * _moms.nu02 * _moms.nu31 + sq02 * _moms.nu40) / mm00_7;

    _amis[7] = (sq20 * _moms.nu22 * _moms.nu04 - sq20*sq13 - 2 * _moms.nu20 * _moms.nu11 * _moms.nu31 * _moms.nu04
                + 2 * _moms.nu20 * _moms.nu11 * _moms.nu22 * _moms.nu13 + _moms.nu20 * _moms.nu02 * _moms.nu40 * _moms.nu04
                - 2 * _moms.nu20 * _moms.nu02 * _moms.nu31 * _moms.nu13 + _moms.nu20 * _moms.nu02 * sq22 + 4 * sq11 * _moms.nu31 * _moms.nu13
                - 4 * sq11 * sq22 - 2 * _moms.nu11 * _moms.nu02 * _moms.nu40 * _moms.nu13 + 2 * _moms.nu11 * _moms.nu02 * _moms.nu31 * _moms.nu22
                + sq02 * _moms.nu40 * _moms.nu22 - sq02 * sq31) / mm00_10;

    _amis[8] = (sq30 * sq12 * _moms.nu04 - 2 * sq30 * _moms.nu12 * _moms.nu03 * _moms.nu13 + sq30 * sq03 * _moms.nu22
                - 2 * _moms.nu30 * sq21 * _moms.nu12 * _moms.nu04 + 2 * _moms.nu30 * sq21 * _moms.nu03 * _moms.nu13
                + 2 * _moms.nu30 * _moms.nu21 * sq12 * _moms.nu13 - 2 * _moms.nu30 * _moms.nu21 * sq03 * _moms.nu31
                - 2 * _moms.nu30 * cc12 * _moms.nu22 + 2 * _moms.nu30 * sq12 * _moms.nu03 * _moms.nu31 + 4 * sq21 * sq21 * _moms.nu04
                - 2 * cc21 * _moms.nu12 * _moms.nu13 - 2 * cc21 * _moms.nu03 * _moms.nu22 + 3 * sq21 * sq12 * _moms.nu22
                + 2 * sq21 * _moms.nu12 * _moms.nu03 * _moms.nu31 + sq21 * sq03 * _moms.nu40 - 2 * _moms.nu21 * cc12 * _moms.nu31
                - 2 * _moms.nu21 * sq12 * _moms.nu03 * _moms.nu40 + sq12 * sq12 * _moms.nu40) / (mm00_7 * mm00_6);

    _amis[9] = (-_moms.nu50 * _moms.nu50 * _moms.nu05 * _moms.nu05 + 10 * _moms.nu50 * _moms.nu41 * _moms.nu14 * _moms.nu05
                - 4 * _moms.nu50 * _moms.nu32 * _moms.nu23 * _moms.nu05 - 16 * _moms.nu50 * _moms.nu32 * sq14 + 12 * _moms.nu50 * sq23 * _moms.nu14
                - 16 * sq41 * _moms.nu23 * _moms.nu05 - 9 * sq41 * sq14 + 12 * _moms.nu41 * sq32 * _moms.nu05
                + 76 * _moms.nu41 * _moms.nu32 * _moms.nu23 * _moms.nu14 - 48 * _moms.nu41 * _moms.nu23 * sq23
                - 48 * _moms.nu32 * sq32 * _moms.nu14 + 32 * sq32 * sq23) / (mm00_7 * mm00_7);


}
