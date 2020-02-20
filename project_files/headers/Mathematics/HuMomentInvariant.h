#ifndef HUMOMENTINVARIANT_H
#define HUMOMENTINVARIANT_H

#include "utils.h"

class HuMomentInvariant
{
public:
    HuMomentInvariant();
    ~HuMomentInvariant();

    HuMomentInvariant(Moments &moms);

    //!Hu Moments elements
    double hu_01, hu_02, hu_03, hu_04 , hu_05, hu_06, hu_07, hu_08;

};

#endif // HUMOMENTINVARIANT_H
