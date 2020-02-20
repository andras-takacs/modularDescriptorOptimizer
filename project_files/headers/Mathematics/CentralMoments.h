#ifndef CENTRALMOMENTS_H
#define CENTRALMOMENTS_H

#include "utils.h"

class CentralMoments
{
public:
    CentralMoments();
    ~CentralMoments();

    double cent_cy, cent_cx, inv_mu00, mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03, mu40, mu31, mu22, mu13, mu04
    , mu50, mu41, mu32, mu23, mu14, mu05;

    double nu00, nu10, nu01, nu20, nu11, nu02, nu30, nu21, nu12, nu03, nu40, nu31, nu22, nu13, nu04
    , nu50, nu41, nu32, nu23, nu14, nu05;

    CentralMoments(RawMoments& _raw_moments);

    CentralMoments(cv::Moments& _cv_moments);

    void calculateCentralMoments(RawMoments& raw_moments,  double* _cent_mom);
};

#endif // CENTRALMOMENTS_H
