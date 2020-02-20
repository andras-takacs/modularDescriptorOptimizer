// -*- c++ -*-
// Copyright 2015 Augmented Technology
//
// MomentsCounter.h
//
// Defines the AffineMomentInvariant class
//
//Calculates the Invariants
//
//
#ifndef MOMENTSCOUNTER
#define MOMENTSCOUNTER

#include "utils.h"

using namespace cv;
using namespace std;


namespace MomentsCounter {

void calculateHighCentralMoment(cv::Mat& _source_im, double& moment_m00, double& moment_m01, double& moment_m10, CentralMoments& c_moments);

}



#endif // MOMENTSCOUNTER

