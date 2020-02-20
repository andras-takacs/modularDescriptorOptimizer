#include "Mathematics/MomentsCounter.h"
#include "utils.h"



void MomentsCounter::calculateHighCentralMoment(cv::Mat &_source_im, double& moment_m00, double& moment_m01, double& moment_m10, CentralMoments& c_moments){

    cv::Size size = _source_im.size();
    int x, y;
    double mus[11] = {0,0,0,0,0,0,0,0,0};
    double _inv_m00 = 1/moment_m00;
    double cx = moment_m10*_inv_m00, cy = moment_m01*_inv_m00;

    for( y = 0; y < size.height; ++y)
    {
        //        const double* ptr = (const double*)(_source_im.data + y*_source_im.step);
        double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;


        for( x = 0; x < size.width; ++x )
        {
            //            double p = ptr[x], xp = x * p, xxp, xxxp;
            double p = _source_im.at<uchar>(y,x), xc = (x-cx) ,xp = xc * p, xxp, xxxp, xxxxp;


            x0 += p;
            x1 += xp;
            xxp = xp * xc;
            x2 += xxp;
            xxxp = xxp * xc;
            x3 += xxxp;
            xxxxp = xxxp * xc;
            x4 += xxxxp;
            x5 += xxxxp * xc;

            //            if(x==125)
            //            std::cout<<"ptr: "<<p<<std::endl;

        }

        double yc = (y - cy), py = yc * x0, sy = yc*yc, ssy = sy*yc, sssy = ssy*yc;

        mus[10] += py * sssy;      // mu05
        mus[9]  += x1 * sssy;      // mu14
        mus[8]  += x2 * ssy;       // mu23
        mus[7]  += x3 * sy;        // mu32
        mus[6]  += x4 * y;         // mu41
        mus[5]  += x5;             // mu50
        mus[4]  += py * ssy;       // mu04
        mus[3]  += x1 * ssy;       // mu13
        mus[2]  += x2 * sy;        // mu22
        mus[1]  += x3 * yc;        // mu31
        mus[0]  += x4;             // mu40

    }

    c_moments.cent_cx = cx;
    c_moments.cent_cy = cy;

    c_moments.mu40 = mus[0];
    c_moments.mu31 = mus[1];
    c_moments.mu22 = mus[2];
    c_moments.mu13 = mus[3];
    c_moments.mu04 = mus[4];
    c_moments.mu50 = mus[5];
    c_moments.mu41 = mus[6];
    c_moments.mu32 = mus[7];
    c_moments.mu23 = mus[8];
    c_moments.mu14 = mus[9];
    c_moments.mu05 = mus[10];

    double inv_sqrt_m00 = std::sqrt(std::abs(_inv_m00));
    double s2 = _inv_m00*_inv_m00,
           s4 = s2*_inv_m00,
           s5 = s4*inv_sqrt_m00;


    c_moments.nu40 = mus[0]*s4;
    c_moments.nu31 = mus[1]*s4;
    c_moments.nu22 = mus[2]*s4;
    c_moments.nu13 = mus[3]*s4;
    c_moments.nu04 = mus[4]*s4;
    c_moments.nu50 = mus[5]*s5;
    c_moments.nu41 = mus[6]*s5;
    c_moments.nu32 = mus[7]*s5;
    c_moments.nu23 = mus[8]*s5;
    c_moments.nu14 = mus[9]*s5;
    c_moments.nu05 = mus[10]*s5;

}
