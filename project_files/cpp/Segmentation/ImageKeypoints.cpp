#include "Segmentation/ImageKeypoints.h"

ImageKeypoints::ImageKeypoints()
{

}

ImageKeypoints::~ImageKeypoints()
{

}

ImageKeypoints::ImageKeypoints(const int& _keyPointType, Mat& _sample_image, const int& _image_margin, int _radius){

    keyPointType = _keyPointType;
    imageWidth = _sample_image.cols;
    imageHeight = _sample_image.rows;
    keyPointRadius = _radius;
    imageMargin = _image_margin;


}

void ImageKeypoints::calculateCoordinates(Images& im_in){


    if (!keypoints.empty()){

        keypoints.clear();
    }

    //===============================================RANDOM keypoints=======================================
    if(keyPointType==RANDOM_KP){

        const int N = 200;
        keypoints.reserve(N);
        for ( int i = 0; i < N; ++i )
        {
            double x=(rand()/(double)RAND_MAX) * imageWidth;
            double y=(rand()/(double)RAND_MAX) * imageHeight;
            if(x<keyPointRadius) x=x+keyPointRadius;
            if(y<keyPointRadius) y=y+keyPointRadius;
            if(x>imageWidth-keyPointRadius) x=x-keyPointRadius;
            if(y>imageHeight-keyPointRadius) y=y-keyPointRadius;
            keypoints.push_back( cv::KeyPoint(x,y,8));
        }
    }
    //===================================FAST keypoints================================================
    if(keyPointType==FAST_KP)
    {
        FAST(im_in.grey_im,keypoints,25,true);

        std::cout<<"FAST keypoints size: "<<keypoints.size()<<std::endl;

    }

    //========================================REMOVING BORDER POINTS=========================================
    for(uint kp=0; kp<keypoints.size();++kp){
        if (keypoints[kp].pt.x<keyPointRadius || keypoints[kp].pt.y<keyPointRadius || keypoints[kp].pt.x>imageWidth-keyPointRadius || keypoints[kp].pt.y>imageHeight-keyPointRadius){
            keypoints.erase(keypoints.begin()+kp);
            kp--;
        }

    }

    //===================================ALL THE IMAGE========================================================

    if(keyPointType==ALL_PIX_KP){

        for (int x=keyPointRadius; x< (imageWidth - keyPointRadius); ++x){
            for (int y=keyPointRadius; y < (imageHeight - keyPointRadius); ++y){
                keypoints.push_back( cv::KeyPoint(x,y,8));
            }
        }
    }

    //=====================================PATCH RADIUS DISTANCE====================================================

    if(keyPointType==PATCH_DIST_KP){
        int step = keyPointRadius*2+1;
        //int step = radius+1;
        int first = keyPointRadius+imageMargin;
        int last = keyPointRadius+imageMargin;// - 1;

        for (int x=first+im_in.rot_offset_x; x< (im_in.col_im.cols - last-im_in.rot_offset_x); x=x+step){

            for (int y=first+im_in.rot_offset_y; y < (im_in.col_im.rows - last-im_in.rot_offset_y); y=y+step){
                //std::cout<<"X: "<<x<<"  "<<"Y: "<<y<<std::endl;
                keypoints.push_back( cv::KeyPoint(x,y,8));
            }
        }

        transformedKeypoints = keypoints;
    }

}


//!checks if the new image size is diffretent.
//!If it is true resets the KeyPoints construct values
bool ImageKeypoints::theImageIsDifferent(Mat& _image_in){

    bool isDifferent = false;
    int in_width = _image_in.cols;
    int in_height = _image_in.rows;

    if(imageWidth!=in_width || imageHeight!=in_height){

        imageWidth = in_width;
        imageHeight = in_height;
        isDifferent = true;

    }else{

        isDifferent = false;
    }

    return isDifferent;
}


cv::KeyPoint ImageKeypoints::pointTransform(KeyPoint _point, Mat _aff_transform_mat){

    Mat warp_mat = _aff_transform_mat;

    KeyPoint dest_point(0,0,8);

    double a00,a01,a10,a11,b00,b10;

    a00 = warp_mat.at<double>(0,0);
    a01 = warp_mat.at<double>(0,1);
    a10 = warp_mat.at<double>(1,0);
    a11 = warp_mat.at<double>(1,1);
    b00 = warp_mat.at<double>(0,2);
    b10 = warp_mat.at<double>(1,2);

    dest_point.pt.x = std::round(_point.pt.x*(float)a00 + _point.pt.y*(float)a01 + (float)b00);
    dest_point.pt.y = std::round(_point.pt.x*(float)a10 + _point.pt.y*(float)a11 + (float)b10);

    return dest_point;

}

void ImageKeypoints::transformAllKeypoints(Mat _aff_transform_mat){

    transformedKeypoints.clear();

    for(uint kp_it=0;kp_it<keypoints.size();++kp_it){

        KeyPoint curr_keypoint = keypoints[kp_it];

        KeyPoint point = pointTransform(cv::KeyPoint(curr_keypoint.pt.x,curr_keypoint.pt.y,8),_aff_transform_mat);
        transformedKeypoints.push_back(point);
    }
}

void ImageKeypoints::homographyTransformAllKeypoints(Mat& _homographyMat){

    transformedKeypoints.clear();
    vector<KeyPoint> _keypoints;

    vector<Point2f> kp_points;
    Mat transformed_kps;


    for(uint kp_it=0;kp_it<keypoints.size();++kp_it){

        kp_points.push_back(keypoints[kp_it].pt);
    }

//    std::cout<<"Keypoint size before: "<<keypoints.size()<<std::endl;

    cv::perspectiveTransform(Mat(kp_points),transformed_kps,_homographyMat);

    for (int pt_it=0;pt_it<transformed_kps.rows;++pt_it){

        KeyPoint curr_kp;
        float x =transformed_kps.at<float>(pt_it,0);
        float y = transformed_kps.at<float>(pt_it,1);


        if(x>(float)imageMargin && y>(float)imageMargin && x<((float)imageWidth-(float)imageMargin) && y<((float)imageHeight-(float)imageMargin)){

            curr_kp.pt.x = std::round(transformed_kps.at<float>(pt_it,0));
            curr_kp.pt.y = std::round(transformed_kps.at<float>(pt_it,1));
            curr_kp.size = 8.0f;

            _keypoints.push_back(keypoints[pt_it]);
            transformedKeypoints.push_back(curr_kp);


        }else{

            continue;
        }
    }

    keypoints.clear();
    keypoints = _keypoints;

//    std::cout<<"Keypoint size after: "<<keypoints.size()<<std::endl;

    transformed_kps.release();

}
