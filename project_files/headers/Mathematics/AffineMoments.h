#ifndef AFFINEMOMENTS_H
#define AFFINEMOMENTS_H

#include "utils.h"

using namespace cv;
using namespace std;

class AffineMoments
{
public:
    AffineMoments();
    ~AffineMoments();

    AffineMoments(CentralMoments& _cent_moments);

    //!Affine Invariant Moments elements
    double ami_01,ami_02,ami_03,ami_04,ami_05,ami_06,ami_07,ami_08,ami_09,ami_10;
    //    double amis[10];

    void calculateInvariants(CentralMoments& _moms, double* _amis);

    void calculateAllInvariants(CentralMoments& _moms, double* _amis);

    static double calculateSingleInvariant(CentralMoments& _moms, int numeration_of_invariant);
};

#endif // AFFINEMOMENTS_H
