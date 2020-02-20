#include "Segmentation/KeyInterestPoint.h"

using namespace std;
using namespace cv;

KeyInterestPoint::KeyInterestPoint()
{
    // keypoints;
}

KeyInterestPoint::~KeyInterestPoint()
{

}

void KeyInterestPoint::extractDescriptorsAndLables(const std::string& testOrTrain,
                                                   const std::string& keyPointType,
                                                   const int& descriptorType,
                                                   cv::Mat &descrip,
                                                   cv::Mat &label,
                                                   std::vector<Images> &im,
                                                   int threshold,
                                                   int loop_start,
                                                   int loop_finish,
                                                   const int _radius)
{

    cv::Mat total_desc,total_label;

    for(int t=loop_start;t<loop_finish;++t)
    {
        cv::Mat in,mask,grey;

        //        std::vector<cv::KeyPoint> keypoints;
        in = im.at(t).col_im;
        mask = im.at(t).mask_im;
        grey = im.at(t).grey_im;

        if (!keypoints.empty()){

            keypoints.clear();
        }

        const int radius =_radius;

        //===============================================RANDOM keypoints=======================================
        if(keyPointType=="RAND"){

            const int N = 200;
            keypoints.reserve(N);
            for ( int i = 0; i < N; ++i )
            {
                double x=(rand()/(double)RAND_MAX) * in.cols;
                double y=(rand()/(double)RAND_MAX) * in.rows;
                if(x<radius) x=x+radius;
                if(y<radius) y=y+radius;
                if(x>in.cols-radius) x=x-radius;
                if(y>in.rows-radius) y=y-radius;
                keypoints.push_back( cv::KeyPoint(x,y,8));
            }
        }
        //===================================FAST keypoints================================================
        if(keyPointType=="FAST")
        {
            FAST(grey,keypoints,threshold,true);

            std::cout<<"FAST keypoints size: "<<keypoints.size()<<std::endl;

        }

        //========================================REMOVING BORDER POINTS=========================================
        for(uint kp=0; kp<keypoints.size();++kp){
            if (keypoints[kp].pt.x<radius || keypoints[kp].pt.y<radius || keypoints[kp].pt.x>in.cols-radius || keypoints[kp].pt.y>in.rows-radius){
                keypoints.erase(keypoints.begin()+kp);
                kp--;
            }

        }

        //===================================ALL THE IMAGE========================================================

        if(keyPointType=="ALL"){

            for (int x=radius; x< (in.cols - radius); ++x){
                for (int y=radius; y < (in.rows - radius); ++y){
                    keypoints.push_back( cv::KeyPoint(x,y,8));
                }
            }
        }

        //=====================================PATCH RADIUS DISTANCE====================================================

        if(keyPointType=="PATCH_DIST"){
            int step = radius*2+1;
            //int step = radius+1;
            int last = radius;// - 1;

            for (int x=radius; x< (in.cols - last); x=x+9){
                for (int y=radius; y < (in.rows - last); y=y+step){
                    //std::cout<<"X: "<<x<<"  "<<"Y: "<<y<<std::endl;
                    keypoints.push_back( cv::KeyPoint(x,y,8));
                }
            }

        }

        //=============================DESCRIPTOR EXTRACTOR TYPES=============================================
        Descriptor* descriptors1 = new Descriptor();

        if (descriptorType == SURF_D){

            descriptors1->descriptor_feature(keypoints,in,SURF_D);
        }
        else if (descriptorType == SIFT_D){

            descriptors1->descriptor_feature(keypoints,in,SIFT_D);
        }
        else if (descriptorType == OPP_SURF_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_SURF_D);
        }
        else if (descriptorType == OPP_SIFT_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_SIFT_D);
        }        
        else if (descriptorType == ORB_D){

            descriptors1->descriptor_feature(keypoints,in,ORB_D);
        }
        else if (descriptorType == BRIEF_D){

            descriptors1->descriptor_feature(keypoints,in,BRIEF_D);
        }
        else if (descriptorType == BRISK_D){

            descriptors1->descriptor_feature(keypoints,in,BRISK_D);
        }
        else if (descriptorType == FREAK_D){

            descriptors1->descriptor_feature(keypoints,in,FREAK_D);
        }
        /*else if (descriptorType == OPP_ORB_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_ORB_D);
        }
        else if (descriptorType == OPP_BRIEF_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_BRIEF_D);
        }
        else if (descriptorType == OPP_BRISK_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_BRISK_D);
        }
        else if (descriptorType == OPP_FREAK_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_FREAK_D);
        }*/
        else if(descriptorType==RGB_D){

            descriptors1->descriptor_feature(keypoints,in,RGB_D);   //Descriptor for 2radius+1 wide featurettes
        }
        else if(descriptorType==M_PROJECT_D){

            descriptors1->descriptor_feature(keypoints,in,M_PROJECT_D);   //Descriptor for 2radius+1 wide featurettes
        }
        else if(descriptorType==D_PROJECT_D){

            descriptors1->descriptor_feature(keypoints,in,D_PROJECT_D);   //Descriptor for 2radius+1 wide featurettes
        }


        //=================================Labeling Keypoints==================================
        const int N = keypoints.size();
        cv::Mat labels( N, 1, CV_32FC1 );

        for ( int mi = 0; mi < N; ++mi )
        {
            if(NUMBER_OF_CLASSES==2){


                labels.at<float>(mi) = (mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x ) > 10) ? 1 : 0;
            }
            else if(NUMBER_OF_CLASSES==3){

                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
                if (255 >= pix && pix > 175){
                    labels.at<float>(mi) = 2;
                }else if (175 >= pix && pix > 90){
                    labels.at<float>(mi) = 1;
                }else{
                    labels.at<float>(mi) = 0;
                }

            }else if(NUMBER_OF_CLASSES==4){

                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
                if (255 >= pix && pix > 175){
                    labels.at<float>(mi) = 3;
                }else if (65 >= pix && pix > 45){
                    labels.at<float>(mi) = 2;
                }else if (45 >= pix && pix > 15){
                    labels.at<float>(mi) = 1;
                }else{
                    labels.at<float>(mi) = 0;
                }

            }else if(NUMBER_OF_CLASSES==9){
                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
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
            }else if(NUMBER_OF_CLASSES==9){
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


                Vec3b m_colour = mask.at<Vec3b>(keypoints[mi].pt.y, keypoints[mi].pt.x);
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

        total_desc.push_back(descriptors1->descriptors);
        total_label.push_back(labels);
        //        keypoints=keypoints;
    }
    descrip = total_desc;
    label = total_label;

    if(testOrTrain == "TEST"){
        std::cout<<"Test descriptors size: "<<descrip.rows<<"  Test descriptors cols: "<<descrip.cols<<std::endl;
        std::cout<<"Test labels row "<<label.rows<<" Test labels cols "<<label.cols<<std::endl;
    }
    else if(testOrTrain == "TRAIN"){
        std::cout<<"Train descriptors size: "<<descrip.rows<<"  Train descriptors cols: "<<descrip.cols<<std::endl;
        std::cout<<"Train labels row "<<label.rows<<" Train labels cols "<<label.cols<<std::endl;
    }
}

void KeyInterestPoint::extraction(const int& testOrTrain,
                                  const int& keyPointType,
                                  const int& labelType,
                                  const int& descriptorType,
                                  cv::Mat &descrip,
                                  cv::Mat &label,
                                  const int& train_percent,
                                  const int _radius)
{

    cv::Mat total_desc,total_label;
    int threshold = 25;
    int loop_start=0, loop_finish=0;
    Database im_db;
    im_db.setDataBase(TILDE_DB, UNI_COMPUTER, testOrTrain);

    if (testOrTrain==TRAIN){
        loop_start =0;
//        loop_finish = (int) std::ceil((float)im_db.size*((float)train_percent/(float)100));
        loop_finish = im_db.size;
//        loop_finish = 80;


    }else if (testOrTrain==TEST){
//        if(train_percent>50){
//            loop_start =(int) std::ceil((float)im_db.size*((float)train_percent/(float)100));
//            loop_finish = im_db.size;
//        }
//        else if(train_percent<10){
            loop_start =train_percent;
            loop_finish = train_percent+1;
//        }
    }
    std::cout<<"Loop start: "<<loop_start<<" Loop finish: "<<loop_finish<<std::endl;

    for(int t=loop_start;t<loop_finish;++t)
    {
        cv::Mat in,mask,grey;

        //        std::vector<cv::KeyPoint> keypoints;

        Images im;
        im.getImagesFromDatabase(im_db,t);

        in = im.col_im;
        mask = im.mask_im;
        grey = im.grey_im;

        if (!keypoints.empty()){

            keypoints.clear();
        }

        const int radius =_radius;

        //===============================================RANDOM keypoints=======================================
        if(keyPointType==RANDOM_KP){

            const int N = 200;
            keypoints.reserve(N);
            for ( int i = 0; i < N; ++i )
            {
                double x=(rand()/(double)RAND_MAX) * in.cols;
                double y=(rand()/(double)RAND_MAX) * in.rows;
                if(x<radius) x=x+radius;
                if(y<radius) y=y+radius;
                if(x>in.cols-radius) x=x-radius;
                if(y>in.rows-radius) y=y-radius;
                keypoints.push_back( cv::KeyPoint(x,y,8));
            }
        }
        //===================================FAST keypoints================================================
        if(keyPointType==FAST_KP)
        {
            FAST(grey,keypoints,threshold,true);

            std::cout<<"FAST keypoints size: "<<keypoints.size()<<std::endl;

        }

        //========================================REMOVING BORDER POINTS=========================================
        for(uint kp=0; kp<keypoints.size();++kp){
            if (keypoints[kp].pt.x<radius || keypoints[kp].pt.y<radius || keypoints[kp].pt.x>in.cols-radius || keypoints[kp].pt.y>in.rows-radius){
                keypoints.erase(keypoints.begin()+kp);
                kp--;
            }

        }

        //===================================ALL THE IMAGE========================================================

        if(keyPointType==ALL_PIX_KP){

            for (int x=radius; x< (in.cols - radius); ++x){
                for (int y=radius; y < (in.rows - radius); ++y){
                    keypoints.push_back( cv::KeyPoint(x,y,8));
                }
            }
            std::cout<<"All keypoints size: "<<keypoints.size()<<std::endl;
        }

        //=====================================PATCH RADIUS DISTANCE====================================================

        if(keyPointType==PATCH_DIST_KP){
            int step = radius*2+1;
            //int step = radius+1;
            int last = radius;// - 1;

            for (int x=radius; x< (in.cols - last); x=step){
                for (int y=radius; y < (in.rows - last); y=step){
                    //std::cout<<"X: "<<x<<"  "<<"Y: "<<y<<std::endl;
                    keypoints.push_back( cv::KeyPoint(x,y,8));
                }
            }

        }

        //=============================DESCRIPTOR EXTRACTOR TYPES=============================================
        Descriptor* descriptors1 = new Descriptor();

        if (descriptorType == SURF_D){

            descriptors1->descriptor_feature(keypoints,in,SURF_D);
        }
        else if (descriptorType == SIFT_D){

            descriptors1->descriptor_feature(keypoints,in,SIFT_D);
        }
        else if (descriptorType == OPP_SURF_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_SURF_D);
        }
        else if (descriptorType == OPP_SIFT_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_SIFT_D);
        }
        else if (descriptorType == ORB_D){

            descriptors1->descriptor_feature(keypoints,in,ORB_D);
        }
        else if (descriptorType == BRIEF_D){

            descriptors1->descriptor_feature(keypoints,in,BRIEF_D);
        }
        else if (descriptorType == BRISK_D){

            descriptors1->descriptor_feature(keypoints,in,BRISK_D);
        }
        else if (descriptorType == FREAK_D){

            descriptors1->descriptor_feature(keypoints,in,FREAK_D);
        }
        /*else if (descriptorType == OPP_ORB_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_ORB_D);
        }
        else if (descriptorType == OPP_BRIEF_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_BRIEF_D);
        }
        else if (descriptorType == OPP_BRISK_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_BRISK_D);
        }
        else if (descriptorType == OPP_FREAK_D){

            descriptors1->descriptor_feature(keypoints,in,OPP_FREAK_D);
        }*/
        else if(descriptorType==RGB_D){
            descriptors1->descriptor_feature(keypoints,in,RGB_D,_radius);   //Descriptor for 2radius+1 wide featurettes
        }
        else if(descriptorType==M_PROJECT_D){
            descriptors1->descriptor_feature(keypoints,in,M_PROJECT_D,_radius);   //Descriptor for 2radius+1 wide featurettes
        }
        else if(descriptorType==D_PROJECT_D){

            descriptors1->descriptor_feature(keypoints,in,D_PROJECT_D);   //Descriptor for 2radius+1 wide featurettes
        }


        //=================================Labeling Keypoints==================================
        const int N = keypoints.size();
        cv::Mat labels( N, 1, CV_32FC1 );

        for ( int mi = 0; mi < N; ++mi )
        {
            if(labelType==PROJECT_2_CLASS){

                labels.at<float>(mi) = (float)(mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x ) > 0) ? 1 : 0;
            }
            else if(labelType==PROJECT_3_CLASS){

                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
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

                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
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
                int pix = (int)mask.at<uchar>( keypoints[mi].pt.y, keypoints[mi].pt.x );
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


                Vec3b m_colour = mask.at<Vec3b>(keypoints[mi].pt.y, keypoints[mi].pt.x);
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

        total_desc.push_back(descriptors1->descriptors);
        total_label.push_back(labels);

        descriptors1->descriptors.release();
        labels.release();
        //        keypoints=keypoints;

        in.release();
        mask.release();
        grey.release();
    }
    descrip = total_desc;
    label = total_label;

    total_desc.release();
    total_label.release();
//    keypoints.clear();

    if(testOrTrain == TEST){
        std::cout<<"Test descriptors size: "<<descrip.rows<<"  Test descriptors cols: "<<descrip.cols<<std::endl;
        std::cout<<"Test labels row "<<label.rows<<" Test labels cols "<<label.cols<<std::endl;
    }
    else if(testOrTrain == TRAIN){
        std::cout<<"Train descriptors size: "<<descrip.rows<<"  Train descriptors cols: "<<descrip.cols<<std::endl;
        std::cout<<"Train labels row "<<label.rows<<" Train labels cols "<<label.cols<<std::endl;
    }
}
