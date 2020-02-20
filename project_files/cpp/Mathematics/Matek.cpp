#include "Mathematics/Matek.h"

/* Function to convert decimal ineger to a
 * binary integer.*/
int Matek::decimalToBinary(int dec)
{
    int rem, i=1, binary=0;
    while (dec!=0)
    {
        rem=dec%2;
        dec/=2;
        binary+=rem*i;
        i*=10;
    }
    return binary;
}

/* Function to convert binary integer to a
 * decimal integer.*/
int Matek::binaryToDecimal(int bin)
{
    int decimal=0, i=0, rem;
    while (bin!=0)
    {
        rem = bin%10;
        bin/=10;
        decimal += rem*pow(2,i);
        ++i;
    }
    return decimal;
}

/*Quick version for the power table of 10*/
int Matek::quickPow10(int pow)
{
    static int pow10[10] = {
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000
    };

    return pow10[pow];
}

/* Converts the "binary" vector<int> into an integer for
 * for further conversion to decimal*/
int Matek::binVectorToInt(std::vector<int> _input){

    int returnValue = 0;

    for(int i=_input.size()-1;i>=0;--i){

        int power = (_input.size()-1) - i;

        returnValue += (_input[i]*quickPow10(power));

    }

    return returnValue;

}

/*Multiply pixelwise a 3D image matrix with a 2D transformation matrix:
 *|T1| |M1 M2 M3| |R|
 *|T2|=|M4 M5 M6|x|G|
 *|T3| |M7 M8 M9| |B|
*/
Mat Matek::elemMatrixTransform(cv::Mat& _inputMat, cv::Mat& _transformationMat){

    Mat floatMat,tMat;

    if(_inputMat.depth()!=CV_32F){
        _inputMat.convertTo(_inputMat, CV_32F);
    }else{
        _inputMat.copyTo(floatMat);
    }

    Mat myDataReshaped = floatMat.reshape(1, floatMat.rows*floatMat.cols); // COL*ROWx3 matrix

    transpose(_transformationMat,tMat);

    myDataReshaped = myDataReshaped*tMat;

    myDataReshaped.release();
    tMat.release();

    return floatMat;

}

/* Normalize RGB values at pixel level
 * c1=C1/C1+C2+C3, c2=C2/C1+C2+C3, c3=C3/C1+C2+C3
 * in the case of RGB:
 * r=R/R+G+B, g=G/R+G+B, b=B/R+G+B */
Mat Matek::colorNormalization(cv::Mat& _inputMat){

    std::vector<cv::Mat> channels(3);
    Mat summedMat,normedIm,floatMat;

    if(_inputMat.channels()<3){

        std::cout<<"The matrix has too few layes for normalization!"<<std::endl;
    }

    if(_inputMat.depth()!=CV_32F){
        _inputMat.convertTo(floatMat, CV_32F);
    }else{
        _inputMat.copyTo(floatMat);
    }

    cv::split(floatMat,channels);

    summedMat = channels[0]+channels[1]+channels[2];

    for(int i=0;i<3;++i){

        channels[i] /=summedMat;
        //        channels[i] *=255; //to see the normalized image
    }

    cv::merge(channels,normedIm);

    //    normedIm.convertTo(normedIm,CV_8U);

    summedMat.release();
    floatMat.release();

    return normedIm;

}

/*Opponent color channel calculator with input matrix RGB
 * o_1 = (R-G)/sqrt(2)
 * o_2 = (R+G-2B)/sqrt(6)
 * o_3 = (R+G+B)/sqrt(3)
 */
Mat Matek::opponentColorMat(cv::Mat& _input_rgb_image){

    std::vector<cv::Mat> channels(3),o(3);
    Mat summedMat,opponentIm,floatMat;
    Mat RminG,RplusG,twoB;

    if(_input_rgb_image.depth()!=CV_32F){
        _input_rgb_image.convertTo(floatMat, CV_32F);
    }else{
        _input_rgb_image.copyTo(floatMat);
    }

    cv::split(floatMat,channels);

    RminG = channels[0] - channels[1];
    RplusG = channels[0] + channels[1];
    twoB = channels[2].mul(channels[2]);

    o[0] = RminG / sqrtf(2.0f);
    o[1] = (RplusG - twoB) / sqrtf(6.0f);
    o[2] = (RplusG + channels[0]) / sqrtf(3.0f),


            cv::merge(o,opponentIm);

    //    normedIm.convertTo(normedIm,CV_8U);

    summedMat.release();
    floatMat.release();
    RminG.release();
    RplusG.release();
    twoB.release();

    return opponentIm;


}

/*Calculates Gevers l1, l2, l3 invariant features with input matrix RGB */
Mat Matek::invariantLoneLtwoLthree(Mat& _input_rgb_image){

    std::vector<cv::Mat> channels(3),lChannels(3);

    Mat RminG, RminB, GminB;
    Mat RminGsq, RminBsq, GminBsq;
    Mat floatMat,divisor,returnMat;

    if(_input_rgb_image.depth()!=CV_32F){
        _input_rgb_image.convertTo(floatMat, CV_32F);
    }else{
        _input_rgb_image.copyTo(floatMat);
    }

    cv::split(floatMat,channels);

    RminG = channels[0]-channels[1];
    RminB = channels[0]-channels[2];
    GminB = channels[1]-channels[2];

    RminGsq = RminG.mul(RminG);
    RminG.release();

    RminBsq = RminB.mul(RminB);
    RminB.release();

    GminBsq = GminB.mul(GminB);
    GminB.release();

    divisor = RminGsq + RminBsq + GminBsq;

    lChannels[0]= RminGsq / divisor;
    lChannels[1]= RminBsq / divisor;
    lChannels[2]= GminBsq / divisor;

    cv::merge(lChannels,returnMat);

    RminGsq.release();
    RminBsq.release();
    GminBsq.release();
    floatMat.release();
    divisor.release();

    //    std::cout<<"l1,l2,l3 value at (100,10): "<<returnMat.at<Vec3f>(100,10)<<std::endl;

    return returnMat;

}

/*Calculates Gevers c1, c2, c3 invariant features with input matrix RGB
 * c1 = arctan(R/max{G,B})
 * c2 = arctan(G/max{R,B})
 * c3 = arctan(B/max{R,G})
*/
Mat Matek::invariantConeCtwoCthree(Mat& _input_rgb_image){

    std::vector<cv::Mat> channels(3),cChannels(3);
    int x = _input_rgb_image.cols;
    int y = _input_rgb_image.rows;

    Mat maxRG, maxRB, maxGB;
    Mat floatMat,returnMat;

    if(_input_rgb_image.depth()!=CV_32F){
        _input_rgb_image.convertTo(floatMat, CV_32F);
    }else{
        _input_rgb_image.copyTo(floatMat);
    }

    cv::split(floatMat,channels);

    cv::max(channels[0],channels[1],maxRG);
    cv::max(channels[0],channels[2],maxRB);
    cv::max(channels[1],channels[2],maxGB);

    cChannels[0] = channels[0] / maxGB;
    cChannels[1] = channels[1] / maxRB;
    cChannels[2] = channels[2] / maxRG;

    for(int c_i=0;c_i<3;++c_i){
        for(int x_i=0;x_i<x;++x_i){
            for(int y_i=0;y_i<y;++y_i){

                cChannels[c_i].at<float>(y_i,x_i) = (float) std::atan((double) cChannels[c_i].at<float>(y_i,x_i));

            }
        }
    }


    cv::merge(cChannels,returnMat);

    maxGB.release();
    maxRB.release();
    maxRG.release();
    floatMat.release();

    //    std::cout<<"c1,c2,c3 value at (100,10): "<<returnMat.at<Vec3f>(100,10)<<std::endl;


    return returnMat;

}

/* Geusebroek Invariant feature H
 * H = E^(lambda)/E^(lambda lambda)
 *
 * Geusebroek Invariant feature C
 * C = E^(lambda)/E
 */
Mat Matek::featureHC(Mat& _float_input_rgb_im){

    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> feature_channels(2);
    Mat geusebroekMat, floatMat;
    /*Geusebroek tranformation Matrix
     *
 */
    Mat M_geusebroek = (Mat_<float>(3,3) << 0.06, 0.63, 0.27, 0.3, 0.04, -0.35, 0.34, -0.6, 0.17);

    if(_float_input_rgb_im.depth()!=CV_32F){
        _float_input_rgb_im.convertTo(floatMat, CV_32F);
    }else{
        _float_input_rgb_im.copyTo(floatMat);
    }

    Mat g_res = elemMatrixTransform(floatMat,M_geusebroek);

    cv::split(g_res,channels);

    feature_channels[0] = channels[1]/channels[2];
    feature_channels[1] = channels[1]/channels[0];
    cv::merge(feature_channels,geusebroekMat);

    floatMat.release();

    return geusebroekMat;
}

std::vector<float> Matek::calculateAngleDifferences(std::vector<float>_incoming_angles){

    std::vector<float> angle_differences;

//    float an = 0;
//    float bn = 355.22238;

//    float d = fmod(std::fabs(an - bn),360.0f);
//    float r = d > 180.0f ? 360.0f - d : d;

//    std::cout<<"Angle difference: "<<r<<std::endl;

    int reducer = 0;
    int case_num = 0;
    for(int a_it=0;a_it<5;++a_it){
        for(int b_it=reducer;b_it<5;++b_it){

            if(a_it!=b_it){

                float d = fmod(std::fabs(_incoming_angles[a_it] - _incoming_angles[b_it]),360.0f);
                float r = d > 180.0f ? 360.0f - d : d;

                case_num++;

                angle_differences.push_back(r);
//                std::cout<<"Case "<<case_num<<" is: "<<r<<std::endl;

            }else{
                continue;
            }
        }

        reducer++;
    }

    return angle_differences;

}
