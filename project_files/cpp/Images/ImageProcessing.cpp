
#include "Images/ImageProcessing.h"
#include "utils.h"

using namespace std;
using namespace cv;



void ImageProcessing::loadMultipleImages(std::vector<Images>& im_vec, uint type){

    char name[200];
    uint start_val=0, end_val=0;
    string framesDirectory;
    string fileNameTemplate;
    string maskDirectory;
    //    string file_name;


    fileNameTemplate = "rgb_%02d.pnm";
    framesDirectory=homeDirectory+"brighton/images/";
    if (NUMBER_OF_CLASSES == 2){
        maskDirectory=homeDirectory+"brighton/labels/2_label_mask";
    }else if (NUMBER_OF_CLASSES == 3){
        maskDirectory=homeDirectory+"brighton/labels/3_label_mask";
    }else if (NUMBER_OF_CLASSES == 4){
        maskDirectory=homeDirectory+"brighton/labels/4_label_mask";
    }else if (NUMBER_OF_CLASSES == 9){
        maskDirectory=homeDirectory+"brighton/labels/multichannel_mask";
    }


    if (type==0){
        start_val = number_of_test_images;
        end_val = number_of_images;
        std::cout<<"Start value of train images: "<<start_val<<" End value: "<<start_val+end_val<<std::endl;
    } else if(type==1){
        start_val = 0;
        end_val = number_of_test_images;
        std::cout<<"Start value of test images: "<<start_val<<" End value: "<<start_val+end_val<<std::endl;
    }


    //    if (number_of_images<10){
    //        number_of_test_images=1;
    //    }else{
    //        number_of_test_images = floor(number_of_images/10);
    //    }

    std::vector<std::string> imageData;
    imageData.push_back(framesDirectory);
    imageData.push_back(maskDirectory);
    imageData.push_back((string) name);

    //    {framesDirectory, maskDirectory, name}
    for (uint ti=0; ti<end_val; ++ti)
    {

        sprintf(name, fileNameTemplate.c_str(), ti+start_val);
        //        file_name = (string)name;
        imageData.at(2)=(string)name;
        im_vec.at(ti).loadImages(imageData);

    }
    std::cout<<"Finished reading all the images"<<std::endl;
}


//! Maddern2014 - Illumination Invariant Imaging... paper
//! the invariant colour value calculation
//! I = log (R2) − α log (R1) − (1 − α) log (R3)
cv::Mat ImageProcessing::illuminationInvariant(cv::Mat& _im)
{

    double _alpha = 0.45;
    cv::Mat ii_im(Mat::zeros(_im.size(),CV_8U));
    cv::Mat f_im;
    cv::GaussianBlur(_im,f_im,Size(5,5),0,0);

    std::vector<cv::Mat> rgb_channels;
    //    std::vector<cv::Mat> conv_rgb_channels;
    cv::split(f_im, rgb_channels);



    //    std::cout<<"Im_type: "<<conv_channel.type()<<std::endl;
    for(uint i=0;i<3;++i){
        cv::Mat conv_channel,dest_mat;

        rgb_channels[i].convertTo(conv_channel, CV_32F);

        cv::log(conv_channel,dest_mat);
        rgb_channels[i] = dest_mat;
    }

    //! OpenCV colour channels are BGR not RGB --> channel[1] is Blue

    Mat a,b,c,d;
    a= (Scalar)0.5 + rgb_channels[2];
    b= rgb_channels[1]*_alpha;
    c= rgb_channels[3]*(1-_alpha);

    d = a-b-c;
    //    ii_im =(((Scalar)0.5 + rgb_channels[2]) - (rgb_channels[3]*_alpha) - (rgb_channels[1]*(1-_alpha)));
    ii_im = d;
    double minVal, maxVal;
    cv::minMaxLoc(d, &minVal, &maxVal); //find minimum and maximum intensities

    d.convertTo(ii_im, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

    return ii_im;
}

Mat ImageProcessing::gammaCorrection(cv::Mat& _src, double gamma){

    //    cv::Mat _dest;

    double inverse_gamma = 1.0 / gamma;

    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ )
        ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

    Mat result;
    LUT( _src, lut_matrix, result );

    return result;
}

Mat ImageProcessing::getAffineTransformMat(int _image_rows, int _image_cols, const std::vector<double> &_aff_triplet){

    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat _aff_mat( 2, 3, CV_64F);

    /// Set your 3 points to calculate the  Affine Transform
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( _image_cols - 1, 0 );
    srcTri[2] = Point2f( 0, _image_rows - 1 );

    //    int grow_factor,shrink;

    //    dstTri[0] = Point2f( _image_cols*0.0, _image_rows*0.1 );
    //    dstTri[1] = Point2f( _image_cols*0.95, _image_rows*0.15);
    //    dstTri[2] = Point2f( _image_cols*0.05, _image_rows*0.9 );

    dstTri[0] = Point2f( _image_cols*_aff_triplet[0], _image_rows*_aff_triplet[0] );
    dstTri[1] = Point2f( _image_cols*_aff_triplet[1] - 1, _image_rows*(1-_aff_triplet[1]) );
    dstTri[2] = Point2f( _image_cols*(1-_aff_triplet[2]), _image_rows*_aff_triplet[2]-1 );

    /// Get the Affine Transform
    _aff_mat = getAffineTransform( srcTri, dstTri );

    return _aff_mat;
}

Mat ImageProcessing::replicateBorderForEvaluation(Mat& _img, int _number_of_pixels){

    Mat dst;
    int top, bottom, left, right;
    int borderType;
    Scalar value;
    RNG rng(12345);


    //    top = (int) (0.05*_img.rows); bottom = (int) (0.05*_img.rows);
    //    left = (int) (0.05*_img.cols); right = (int) (0.05*_img.cols);

    top = (int) (_number_of_pixels); bottom = (int) (_number_of_pixels);
    left = (int) (_number_of_pixels); right = (int) (_number_of_pixels);

    borderType = BORDER_REFLECT_101;

    value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );

    copyMakeBorder( _img, dst, top, bottom, left, right, borderType, value );

    return dst;

}

KeyPoint ImageProcessing::pointTransform(KeyPoint _point, Mat _aff_transform_mat){

    Mat warp_mat = _aff_transform_mat;

    KeyPoint dest_point(0,0,8);

    double a00,a01,a10,a11,b00,b10;

    a00 = warp_mat.at<double>(0,0);
    a01 = warp_mat.at<double>(0,1);
    a10 = warp_mat.at<double>(1,0);
    a11 = warp_mat.at<double>(1,1);
    b00 = warp_mat.at<double>(0,2);
    b10 = warp_mat.at<double>(1,2);

    //    std::cout<<"a00: " << a00<<" a01: " << a01<<" a10: " << a10 <<" a11: " << a11 <<" b00: " << b00<<" b10: " << b10<<std::endl;

    dest_point.pt.x = std::round(_point.pt.x*a00 + _point.pt.y*a01 + b00);
    dest_point.pt.y = std::round(_point.pt.x*a10 + _point.pt.y*a11 + b10);

    //    std::cout<<"Transform matrix: "<<warp_mat<<std::endl;

    //    warp_mat.release(dest_point.at<float>(0),dest_point.at<float>(1));

    //    std::cout<<"Keypoint coordinates: " << dest_point.pt.x<<", "<< dest_point.pt.y << std::endl;

    return dest_point;

}


Mat ImageProcessing::lightIntensityChangeAndShift(Mat& _image, double _alpha, int _beta){

    //    double alpha; /**< Simple contrast control */
    //    int beta;  /**< Simple brightness control */


    Mat image = _image;
    Mat new_image = Mat::zeros( image.size(), image.type() );


    /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
    for( int y = 0; y < image.rows; y++ )
    { for( int x = 0; x < image.cols; x++ )
        { for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] =
                        saturate_cast<uchar>( _alpha*( image.at<Vec3b>(y,x)[c] ) + _beta );
            }
        }


    }

    return new_image;
}


Mat ImageProcessing::colorTemperatureChange(Mat& _image, const std::vector<double> &_alpha, int _beta){

    //    double alpha; /**< Simple contrast control */
    //    int beta;  /**< Simple brightness control */


    Mat image = _image;
    Mat new_image = Mat::zeros( image.size(), image.type() );


    /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
    for( int y = 0; y < image.rows; y++ )
    { for( int x = 0; x < image.cols; x++ )
        { for( int c = 0; c < 3; c++ )
            {
                new_image.at<Vec3b>(y,x)[c] =
                        saturate_cast<uchar>( _alpha[c]*( image.at<Vec3b>(y,x)[c] ) + _beta );
            }
        }


    }

    return new_image;
}


Images ImageProcessing::enlargeImageForRotationAndSize(Images& _src, int eval_type){
    //    int theta = _theta;
    Images returnImages;
    int diagonal=0;
    Mat src,src_grey,src_mask;
    src = _src.col_im;
    src_grey = _src.grey_im;
    src_mask = _src.mask_im;
    //    src = imread("../rgb_00.pnm",1);
    //    cout<<endl<<endl<<"Press '+' to rotate anti-clockwise and '-' for clockwise 's' to save" <<endl<<endl;

    if(eval_type==ROTATE){
        diagonal = (int)sqrt(src.cols*src.cols+src.rows*src.rows);
    }else if(eval_type==SIZE){
        diagonal = (int)src.cols*1.5;
    }
    int newWidth = diagonal;
    int newHeight =diagonal;

    int offsetX = (newWidth - src.cols) / 2;
    int offsetY = (newHeight - src.rows) / 2;
    Mat targetMat(newWidth, newHeight, src.type());
    Mat targetGrey(newWidth, newHeight, src_grey.type());
    Mat targetMask(newWidth, newHeight, src_mask.type());
    //    Point2f src_center(targetMat.cols/2.0F, targetMat.rows/2.0F);

    returnImages.rot_offset_x = offsetX;
    returnImages.rot_offset_y = offsetY;

    src.copyTo(targetMat.rowRange(offsetY, offsetY + src.rows).colRange(offsetX, offsetX + src.cols));
    src_grey.copyTo(targetGrey.rowRange(offsetY, offsetY + src_grey.rows).colRange(offsetX, offsetX + src_grey.cols));
    src_mask.copyTo(targetMask.rowRange(offsetY, offsetY + src_mask.rows).colRange(offsetX, offsetX + src_mask.cols));

    returnImages.col_im = targetMat;
    returnImages.grey_im = targetGrey;
    returnImages.mask_im = targetMask;

    return returnImages;

}

Mat ImageProcessing::normalizeWithLocalMax(Mat& _image){

    //    std::vector<cv::Mat> splitChannels;
    //    cv::Mat doubleImage;

    ////    std::cout<<"Normalized size: "<<_image.cols<<"x"<<_image.rows<<std::endl;

    //    cv::Mat _returnImage(Mat::zeros(_image.size(),CV_32FC3));
    //    cv::Mat _pre_return_image(Mat::zeros(_image.size(),CV_64FC3));

    //    cv::split(_image, splitChannels);

    //    for(int i=0;i<3;++i){

    //        cv::Mat currentChannel = splitChannels[i];
    //        currentChannel.convertTo(currentChannel,CV_64F);
    ////        std::cout<<"Begin Channel: "<<currentChannel<<std::endl;
    //        double maxVal,minVal ,averageOfPixels;
    //        cv::minMaxLoc(currentChannel, &minVal, &maxVal);

    ////        currentChannel = cv::abs(currentChannel - maxVal);

    //        double range = maxVal - minVal;

    ////        std::cout<<"Range: "<<range<<std::endl;

    //        if (range == 0)
    //            currentChannel = (currentChannel - minVal);//the score is a constant value, returns zero
    //        else
    //            currentChannel = (currentChannel - minVal) / range;

    ////        std::cout<<"Middle Channel: "<<currentChannel<<std::endl;

    ////        averageOfPixels = cv::sum(currentChannel)[0]/(double)(currentChannel.cols*currentChannel.rows);
    ////        averageOfPixels = cv::sum(currentChannel)[0]/(currentChannel.cols*currentChannel.rows);

    ////        currentChannel = currentChannel /(int) averageOfPixels;

    //        splitChannels[i] = currentChannel;

    ////        std::cout<<"Normal Channel: "<<currentChannel<<std::endl;
    //    }

    //    cv::merge(splitChannels,_pre_return_image);

    //    _pre_return_image.convertTo(_returnImage,CV_32FC3);

    //    return _returnImage;


    //!WITH ERROR IN THE CODE
    std::vector<cv::Mat> splitChannels;

    cv::Mat _returnImage(Mat::zeros(_image.size(),CV_64FC3));

    cv::split(_image, splitChannels);

    for(int i=0;i<3;++i){

        cv::Mat currentChannel = splitChannels[i];

        double maxVal,minVal, averageOfPixels;
        cv::minMaxLoc(currentChannel, &minVal, &maxVal);

        currentChannel = cv::abs(currentChannel - maxVal);

        averageOfPixels = cv::sum(currentChannel)[0]/(double)(currentChannel.cols*currentChannel.rows);

        currentChannel = currentChannel / averageOfPixels;


        splitChannels[i] = currentChannel;
    }

    cv::merge(splitChannels,_returnImage);

    return _returnImage;

}

Mat ImageProcessing::normalizeImage(Mat& _image){

    cv::Mat _returnImage;

    std::vector<cv::Mat> splitChannels;

    cv::split(_image, splitChannels);

    for(int i=0;i<3;++i){

        cv::Mat currentChannel = splitChannels[i];

        double maxVal,minVal;
        cv::minMaxLoc(currentChannel, &minVal, &maxVal);


        cv::normalize(currentChannel,splitChannels[i],minVal,maxVal,NORM_MINMAX,CV_8UC1);
    }

    cv::merge(splitChannels,_returnImage);

    return _returnImage;
}

Mat ImageProcessing::calculateEigenValues(Mat& _image){

    /// Set some parameters
    int blockSize = 9; int apertureSize = 5;

    /// My Harris matrix -- Using cornerEigenValsAndVecs
    Mat eigenMatrix = Mat::zeros( _image.size(), CV_32FC(6) );
    //      Mc = Mat::zeros( _image.size(), CV_32FC1 );

    cornerEigenValsAndVecs( _image, eigenMatrix, blockSize, apertureSize, BORDER_DEFAULT );

    //      /* calculate Mc */
    //      for( int j = 0; j < src_gray.rows; j++ )
    //         { for( int i = 0; i < src_gray.cols; i++ )
    //              {
    //                float lambda_1 = myHarris_dst.at<Vec6f>(j, i)[0];
    //                float lambda_2 = myHarris_dst.at<Vec6f>(j, i)[1];
    //                Mc.at<float>(j,i) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
    //              }
    //         }


    return eigenMatrix;
}

Mat ImageProcessing::equalizeIntensity(const Mat& inputImage)
{
    Mat result;

    if(inputImage.channels() == 1){

        equalizeHist( inputImage, result );

    }
    else if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);


        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);


    }
    return result;
}

std::vector<cv::Mat> ImageProcessing::calculateGradientAngles(Mat& _inputImg, int _grad_detector){

    Mat _img;
    _inputImg.copyTo(_img);
    std::vector<Mat>image_channels(3),angle_channels(3);

    split(_img,image_channels);


    for(int ch_it=0;ch_it<3;++ch_it){
        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_32F;

        if(_grad_detector==SOBEL){
            /// Gradient X
            Sobel( image_channels[ch_it], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

            /// Gradient Y
            Sobel( image_channels[ch_it], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

        }else if(_grad_detector==SHARR){
            /// Gradient X
            Scharr( image_channels[ch_it], grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );

            /// Gradient Y
            Scharr( image_channels[ch_it], grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        }


        Mat angle;

        phase(grad_x,grad_y,angle,true);

        angle.copyTo(angle_channels[ch_it]);

        angle.release();
        grad_x.release();
        grad_y.release();
    }


    //    cv::normalize(angle,angle,0.0,1.0,NORM_MINMAX,CV_32F);

    //    std::cout<<angle<<std::endl;

    //    double min,max;
    //    cv::minMaxIdx(angle,&min,&max);

    //    std::cout<<"Angle min: "<<min<<" max: "<<max<<std::endl;



    return angle_channels;



}


