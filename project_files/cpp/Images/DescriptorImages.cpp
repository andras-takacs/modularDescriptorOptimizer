#include "Images/DescriptorImages.h"



DescriptorImages::DescriptorImages()
{
    blured_rgb_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    blured_hsv_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    blured_gray_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
    dist_trans_img = cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0));
}

DescriptorImages::~DescriptorImages()
{
    blured_rgb_img.release();
    blured_hsv_img.release();
    blured_gray_img.release();
    dist_trans_img.release();
}


//!Preprocessing function for descriptor image
//!Here the image is in RGB not BGR ==> converted in loading
void DescriptorImages::prepareImageForModularDescriptor(const Mat &_img,std::shared_ptr<MDADescriptor> _descriptor){

    currDescriptor = _descriptor;

    double g_sigma = currDescriptor->getBlurSigma();
    int kernel_size = currDescriptor->getGaussianKernelSize();
    int canny_kernel_size = currDescriptor->getCannyKernelSize();
    int canny_treshold = currDescriptor->getCannyThreshold();

    //Step 1. - Histogram equalization on the image
    Mat eq_im = ImageProcessing::equalizeIntensity(_img);

    //Step 2. - Bluring the image to remove noise
    cv::GaussianBlur(eq_im,blured_rgb_img,Size(kernel_size,kernel_size),g_sigma,g_sigma,BORDER_REFLECT);

    //Step 3. - convert image to a 32 bit matrix for other color spaces
    //image turned to 32 bit image for to have 360 degrees in Hue
    blured_rgb_img.convertTo(float_img, CV_32F);
    float_img *=1./255;
    cv::cvtColor(float_img,blured_hsv_img,CV_RGB2HSV);

    //Step 4. - Image conversion to color space specified by genome
    Mat pre_converted_img =  genomeSpecifiedSpaceConversion(currDescriptor->getColorChannel(),float_img);
    converted_img = normalizeImage(pre_converted_img);
    pre_converted_img.release();

    //Step 5. - Convert the RGB image to grayscale image for distance transform
    cv::cvtColor(blured_rgb_img,blured_gray_img,CV_RGB2GRAY);

    //Step 6. - Calculate canny edge detector for Distance Transform
    Mat edges_img;
    cv::Canny(blured_gray_img,edges_img,canny_treshold,canny_treshold*3,canny_kernel_size,true);
    Mat sub_mat = Mat::ones(edges_img.size(), edges_img.type())*255;

    //Step 7. - Create Distance Transform image
    //subtract the original matrix by sub_mat to give the negative output new_image
    cv::subtract(sub_mat, edges_img, edges_img);
    //calculate the distance transform
    distanceTransform(edges_img, dist_trans_img, CV_DIST_L2, 3);
    //normalize it
    normalize(dist_trans_img, dist_trans_img, 0.0, 1.0, NORM_MINMAX);
    edges_img.release();

    //Step 8. - Calculate Gevers3L and 3c matrices and Geusebroek features
    gevers3l = Matek::invariantLoneLtwoLthree(float_img);
    gevers3c = Matek::invariantConeCtwoCthree(float_img);
    geusebroekHC = Matek::featureHC(float_img);

    //Step 9. Normalize RGB image ==> RGB-->rgb
    normRGB = Matek::colorNormalization(blured_rgb_img);

    //STEP 10. Calculate gradient angles at pixels
    gradient_angles = ImageProcessing::calculateGradientAngles(blured_rgb_img,currDescriptor->getGradientDetector());


}

void DescriptorImages::prepareImageForTestDescriptor(const Mat &_img){

//    currDescriptor = _descriptor;

    double g_sigma = 1.5;
    int kernel_size = 7;
    int canny_kernel_size = 5;
    int canny_treshold = 65;
    int colorSpace = 3;

    //Step 1. - Histogram equalization on the image
    Mat eq_im = ImageProcessing::equalizeIntensity(_img);

    //Step 2. - Bluring the image to remove noise
    cv::GaussianBlur(eq_im,blured_rgb_img,Size(kernel_size,kernel_size),g_sigma,g_sigma,BORDER_REFLECT);

    //Step 3. - convert image to a 32 bit matrix for other color spaces
    //image turned to 32 bit image for to have 360 degrees in Hue
    blured_rgb_img.convertTo(float_img, CV_32F);
    float_img *=1./255;
    cv::cvtColor(float_img,blured_hsv_img,CV_RGB2HSV);

    //Step 4. - Image conversion to color space specified by genome
    Mat pre_converted_img =  genomeSpecifiedSpaceConversion(colorSpace,float_img);
    converted_img = normalizeImage(pre_converted_img);
    pre_converted_img.release();


    //Step 5. - Convert the RGB image to grayscale image for distance transform
    cv::cvtColor(blured_rgb_img,blured_gray_img,CV_RGB2GRAY);

    //Step 6. - Calculate canny edge detector for Distance Transform
    Mat edges_img;
    cv::Canny(blured_gray_img,edges_img,canny_treshold,canny_treshold*3,canny_kernel_size,true);
    Mat sub_mat = Mat::ones(edges_img.size(), edges_img.type())*255;

    //Step 7. - Create Distance Transform image
    //subtract the original matrix by sub_mat to give the negative output new_image
    cv::subtract(sub_mat, edges_img, edges_img);
    //calculate the distance transform
    distanceTransform(edges_img, dist_trans_img, CV_DIST_L2, 3);
    //normalize it
    normalize(dist_trans_img, dist_trans_img, 0.0, 1.0, NORM_MINMAX);
    edges_img.release();

    //Step 8. - Calculate Gevers3L and 3c matrices and Geusebroek features
    gevers3l = Matek::invariantLoneLtwoLthree(float_img);
    gevers3c = Matek::invariantConeCtwoCthree(float_img);
    geusebroekHC = Matek::featureHC(float_img);

    //Step 9. Normalize RGB image ==> RGB-->rgb
    normRGB = Matek::colorNormalization(blured_rgb_img);

    //STEP 10. Calculate gradient angles at pixels
    gradient_angles = ImageProcessing::calculateGradientAngles(blured_rgb_img,SOBEL);


}


void DescriptorImages::prepareImageForDescriptor(const Mat &_img){


    //histogram equalization on the image
    Mat eq_im = ImageProcessing::equalizeIntensity(_img);

    cv::GaussianBlur(eq_im,blured_rgb_img,Size(11,11),30,30,BORDER_CONSTANT);

    //===========NB:Here the image in RGB not BGR==============

    Mat blured_rgb_img_32;
    blured_rgb_img.convertTo(blured_rgb_img_32, CV_32F);
    cv::cvtColor(blured_rgb_img_32,blured_hsv_img,CV_RGB2HSV);
    //    cv::cvtColor(bulred_rgb_img,bulred_hsv_img,CV_RGB2HSV);
    //=================================================================================================

    //    //===============Image comes in as RGB=============================================================
    //
    //=================================================================================================
    cv::cvtColor(blured_rgb_img,blured_gray_img,CV_RGB2GRAY);

    //calculate eigen vectors
    //    eigen_vals = ImageProcessing::calculateEigenValues(blured_gray_img);

    //Calculate canny edge detector
    Mat edges_img;
    cv::Canny(blured_gray_img,edges_img,0,255,5,true);
    Mat sub_mat = Mat::ones(edges_img.size(), edges_img.type())*255;

    //subtract the original matrix by sub_mat to give the negative output new_image
    cv::subtract(sub_mat, edges_img, edges_img);
    //calculate the distance transform
    distanceTransform(edges_img, dist_trans_img, CV_DIST_L2, 3);
    //normalize it
    normalize(dist_trans_img, dist_trans_img, 0.0, 1.0, NORM_MINMAX);
    edges_img.release();

}

Mat DescriptorImages::genomeSpecifiedSpaceConversion(int _genome, Mat& _img){

    int _color_conv =0;
    Mat returnMatrix;

    switch (_genome)
    {
    case RGB_CH:
        _color_conv = 0;
        returnMatrix = Matek::colorNormalization(blured_rgb_img);
        break;
    case Lab_CH:
        _color_conv = CV_RGB2Lab;
        break;
    case Luv_CH:
        _color_conv = CV_RGB2Luv;
        break;
    case XYZ_CH:
        _color_conv = CV_RGB2XYZ;
        break;
    case HSV_CH:
        _color_conv = CV_RGB2HSV;
        break;
    case HLS_CH:
        _color_conv = CV_RGB2HLS;
        break;
    case YCrCb_CH:
        _color_conv = CV_RGB2YCrCb;
        break;
    case OPP_CH:
        _color_conv = 0;
        returnMatrix = Matek::opponentColorMat(_img);
        break;
    default:
        _color_conv = 0;
        returnMatrix = Matek::colorNormalization(blured_rgb_img);
        break;
    }

    if(_color_conv>0){

        cv::cvtColor(_img,returnMatrix,_color_conv);
    }

    return returnMatrix;
}

Mat DescriptorImages::normalizeImage(Mat& _img){

    Mat returnImage;

    vector<Mat>im_channels(3);

    cv::split(_img,im_channels);

    for (int i=0;i<3;++i){

        normalize(im_channels[i],im_channels[i],0.0, 1.0, NORM_MINMAX,CV_32F);

//        double min,max;
//        cv::minMaxIdx(im_channels[i],&min,&max);

//        std::cout<<"Luv min: "<<min<<" max: "<<max<<std::endl;

    }

    cv::merge(im_channels,returnImage);

    return returnImage;

}
