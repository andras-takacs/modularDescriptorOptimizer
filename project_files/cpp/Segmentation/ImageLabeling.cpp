#include "Segmentation/ImageLabeling.h"

ImageLabeling::ImageLabeling()
{
    labelType = PROJECT_4_CLASS;
}

ImageLabeling::ImageLabeling(const int& _labelType)
{
    labelType=_labelType;
}

ImageLabeling::~ImageLabeling()
{

}

Mat ImageLabeling::labelingImage(Images& _im_in, vector<KeyPoint>& _keypoints){

    const int N = _keypoints.size();
    Mat mask = _im_in.mask_im;
    cv::Mat labels( N, 1, CV_32FC1 );

    for ( int mi = 0; mi < N; ++mi )
    {
        if(labelType==PROJECT_2_CLASS){


            labels.at<float>(mi) = (mask.at<uchar>( _keypoints[mi].pt.y, _keypoints[mi].pt.x ) > 0) ? 1 : 0;
        }
        else if(labelType==PROJECT_3_CLASS){

            int pix = (int)mask.at<uchar>( _keypoints[mi].pt.y, _keypoints[mi].pt.x );
            if (255 >= pix && pix > 175){
                labels.at<float>(mi) = 2;
            }else if (175 >= pix && pix > 90){
                labels.at<float>(mi) = 1;
            }else{
                labels.at<float>(mi) = 0;
            }

        }else if(labelType==PROJECT_4_CLASS){

            //! color codes for labels are (in greyscale):
            //! 0 - else = 0
            //! 1 - door & window = 25
            //! 2 - roof = 50
            //! 3 - wall = 255

            int pix = (int)mask.at<uchar>( _keypoints[mi].pt.y, _keypoints[mi].pt.x );
            if (255 >= pix && pix > 175){
                labels.at<float>(mi) = 3;
            }else if (65 >= pix && pix > 45){
                labels.at<float>(mi) = 2;
            }else if (45 >= pix && pix > 15){
                labels.at<float>(mi) = 1;
            }else{
                labels.at<float>(mi) = 0;
            }

        }else if(labelType==PROJECT_9_CLASS){
            int pix = (int)mask.at<uchar>( _keypoints[mi].pt.y, _keypoints[mi].pt.x );
            if (255 >= pix && pix > 240){
                labels.at<float>(mi) = 3;
            }else if (240 >= pix && pix > 215){
                labels.at<float>(mi) = 4;
            }else if (215 >= pix && pix > 185){
                labels.at<float>(mi) = 5;
            }else if (185 >= pix && pix > 165){
                labels.at<float>(mi) = 6;
            }else if (165 >= pix && pix > 135){
                labels.at<float>(mi) = 7;
            }else if (135 >= pix && pix > 115){
                labels.at<float>(mi) = 8;
            }else if (65 >= pix && pix > 45){
                labels.at<float>(mi) = 2;
            }else if (45 >= pix && pix > 15){
                labels.at<float>(mi) = 1;
            }else if (15 >= pix && pix >= 0){
                labels.at<float>(mi) = 0;
            }else{
                labels.at<float>(mi) = 0;
            }
            //std::cout<<"Label values at "<<mi<<" : "<< labels.at<float>(mi)<<" with pix: "<<pix<< std::endl;
        }else if(labelType==LABELME_FACADE_9_CLASS){
            //! color codes for labels are (in RGB):
            //! 0 - various = 0:0:0
            //! 1 - building = 128:0:0
            //! 2 - door = 128:128:0
            //! 3 - window = 0:0:128
            //! 4 - car = 128:0:128
            //! 5 - pavement = 128:128:128
            //! 6 - road = 128:64:0
            //! 7 - sky = 0:128:128
            //! 8 - vegetation = 0:128:0


            Vec3b m_colour = mask.at<Vec3b>(_keypoints[mi].pt.y, _keypoints[mi].pt.x);
            int blue = m_colour.val[0];
            int green = m_colour.val[1];
            int red = m_colour.val[2];

            if(red==0 && green==0 && blue==0){
                labels.at<float>(mi) = 0;
            }else if(red==128 && green==0 && blue==0){
                labels.at<float>(mi) = 1;
            }else if(red==128 && green==128 && blue==0){
                labels.at<float>(mi) = 2;
            }else if(red==0 && green==0 && blue==128){
                labels.at<float>(mi) = 3;
            }else if(red==128 && green==0 && blue==128){
                labels.at<float>(mi) = 4;
            }else if(red==128 && green==128 && blue==128){
                labels.at<float>(mi) = 5;
            }else if(red==128 && green==64 && blue==0){
                labels.at<float>(mi) = 6;
            }else if(red==0 && green==128 && blue==128){
                labels.at<float>(mi) = 7;
            }else if(red==0 && green==128 && blue==0){
                labels.at<float>(mi) = 8;
            }
        }
    }

    return labels;

}

