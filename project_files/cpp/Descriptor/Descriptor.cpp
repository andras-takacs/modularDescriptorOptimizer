#include "Descriptor/Descriptor.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Descriptor::Descriptor()
{
    //descriptors;
}

Descriptor::~Descriptor()
{

}

void Descriptor::descriptor_feature (vector<KeyPoint> &keypoint,
                                     Mat const&image,
                                     const int& descriptorType,
                                     int _radius){


    uint _patch_size = _radius*2+1;

    Mat _raw_descriptor;


    //!===========================
    //! GREY IMAGE DESCRIPTORS ===
    //!===========================

    if (descriptorType == SURF_D){
        //cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("SURF");
        Ptr<SURF> descriptorExtractor = SURF::create( 400 );
        descriptorExtractor->detectAndCompute(image, noArray(), keypoint, _raw_descriptor, true); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == SIFT_D){
        //cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("SIFT");
        Ptr<SIFT> descriptorExtractor = SIFT::create();
        descriptorExtractor->detectAndCompute( image, noArray(), keypoint, _raw_descriptor, true); //OpenCV built in descriptor extractor

        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == ORB_D){
        //cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("ORB");
        Ptr<ORB> descriptorExtractor = ORB::create();
        descriptorExtractor->detectAndCompute( image, noArray(), keypoint, _raw_descriptor, true ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == BRIEF_D){
        //cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("BRIEF");
        cv::Ptr<BriefDescriptorExtractor> descriptorExtractor = BriefDescriptorExtractor::create();
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == BRISK_D){
        cv::Ptr<BRISK> descriptorExtractor = BRISK::create();
        descriptorExtractor->detectAndCompute( image, noArray(), keypoint, _raw_descriptor, true ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == FREAK_D){
        //cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("FREAK");
        //std::cout<<"Descriptor is FREAK!"<<std::endl;
        Ptr<FREAK> descriptorExtractor = FREAK::create();
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == LATCH_D){
        //std::cout<<"Descriptor is LATCH!"<<std::endl;
        Ptr<LATCH> descriptorExtractor = LATCH::create();
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        //std::cout<<"Number of keypoints: "<<keypoint.size()<<", Descriptor size: "<<_raw_descriptor.dims<<std::endl;
        isFloatMatrix(_raw_descriptor);
    }
    //!===========================
    //! OPPONENT COULOUR CHANNELS=
    //!===========================
    /*else if (descriptorType == OPP_SIFT_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SiftDescriptorExtractor()));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == OPP_SURF_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor()));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == OPP_ORB_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new OrbDescriptorExtractor()));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == OPP_BRIEF_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new BriefDescriptorExtractor()));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == OPP_BRISK_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(cv::DescriptorExtractor::create("BRISK"));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }
    else if (descriptorType == OPP_FREAK_D){
        cv::Ptr<cv::OpponentColorDescriptorExtractor>descriptorExtractor
                = new OpponentColorDescriptorExtractor(cv::DescriptorExtractor::create("FREAK"));
        descriptorExtractor->compute( image, keypoint, _raw_descriptor ); //OpenCV built in descriptor extractor
        isFloatMatrix(_raw_descriptor);
    }*/


    //!===========================
    //! PROJECT DESCRIPTORS ======
    //!===========================

    else if (descriptorType==RGB_D){

        rgbDescriptor(image,keypoint,_patch_size,false);

    }

    else if(descriptorType==M_PROJECT_D){

        projectDescriptor(image,keypoint,_patch_size,false);
    }

    else if(descriptorType==D_PROJECT_D){

        doctorateDescriptor(image,keypoint,_patch_size,true);
    }

    _raw_descriptor.release();


}

void Descriptor::rgbDescriptor(cv::Mat const& _img, std::vector<cv::KeyPoint> const&keypoint, uint _patch_size, bool seeSize) {

    uint N = _patch_size*_patch_size*3;
    uint radius = (_patch_size-1)/2;
    Mat _rgb_descriptor=Mat::zeros(keypoint.size(),N,CV_32FC1);

    //!ADD the RGB values to the desscriptor in (R,G,B,R,G...) order

    Mat rgb;
    rgb =_img;
    for (uint kp=0; kp<keypoint.size();++kp){
        float y=keypoint[kp].pt.y;
        float x=keypoint[kp].pt.x;
        int it = 0;
        for (uint i=0; i<_patch_size; ++i){
            for (uint j=0; j<_patch_size; ++j){
                Vec3b intensity = rgb.at<Vec3b>(y-radius+i,x-radius+j);
                for(int ch=0; ch<rgb.channels();++ch){
                    _rgb_descriptor.at<float>(kp,it)=(float)intensity[ch];
                    ++it;
                }
            }
        }
    }

    if (seeSize)
        std::cout<<"RGB DECRIPTOR SIZE: WIDTH: "<<_rgb_descriptor.cols<<" HEIGHT: "<<_rgb_descriptor.rows<<std::endl;

    descriptors = _rgb_descriptor;
}


void Descriptor::projectDescriptor(cv::Mat const& _img, std::vector<cv::KeyPoint> const&keypoint, uint _patch_size,bool seeSize){


    //size of the descriptor vector (2+6+24+(_patch_size*_patch_size))
    //    int N = 113;
    int N = 2+6+24+(_patch_size*_patch_size);
    uint radius = (_patch_size-1)/2;
    int width = _img.cols;
    int height = _img.rows;

    Mat _project_descriptor=Mat::zeros(keypoint.size(),N,CV_32FC1);

    //! Prepare image for descriptor extraction --> bluring, edge extraction, distance transform creation
    DescriptorImages desc_imgs;
    desc_imgs.prepareImageForDescriptor(_img);

    //!=====================================================

    vector<Mat> hsv_channels(3);
    vector<Mat> rgb_channels(3);
    //        desc_imgs.bulred_rgb_img = ImageProcessing::normalizeWithLocalMax(desc_imgs.bulred_rgb_img);
    //        desc_imgs.bulred_hsv_img = ImageProcessing::normalizeWithLocalMax(desc_imgs.bulred_hsv_img);
    cv::split(desc_imgs.blured_rgb_img,rgb_channels);
    cv::split(desc_imgs.blured_hsv_img,hsv_channels);

    //    std::cout<<"Got here!"<<std::endl;

    //calculate at the keypoints the patches and descriptor values
    for (uint kp=0; kp<keypoint.size();++kp){
        int roiWidth = radius*2+1;
        int roiHeight = radius*2+1;
        int roiVertexXCoordinate = keypoint[kp].pt.x - radius;
        int roiVertexYCoordinate = keypoint[kp].pt.y - radius;

        //        if(kp==0)
        //        std::cout<<"First Keypoint: "<<keypoint[kp].pt.x<<", "<<keypoint[kp].pt.y<<std::endl;

        //defines roi
        cv::Rect roi( roiVertexXCoordinate, roiVertexYCoordinate, roiWidth, roiHeight );

        //copies input image in roi
        cv::Mat image_roi_rgb = desc_imgs.blured_rgb_img( roi );
        cv::Mat image_roi_hsv = desc_imgs.blured_hsv_img( roi );

        image_roi_rgb = ImageProcessing::normalizeWithLocalMax(image_roi_rgb);
        //        image_roi_hsv = ImageProcessing::normalizeWithLocalMax(image_roi_hsv);

        cv::Mat image_roi_dist_transf = desc_imgs.dist_trans_img( roi );

        //computes mean over roi
        cv::Scalar avgPixelIntensity_rgb = cv::mean( image_roi_rgb );
        cv::Scalar avgPixelIntensity_hsv = cv::mean( image_roi_hsv );

        //FEED the descriptor with data
        int it =0;

        //The X an Y position (2 val)
        _project_descriptor.at<float>(kp,it)= (float)keypoint[kp].pt.x/(float)width;
        ++it;
        _project_descriptor.at<float>(kp,it)= (float)keypoint[kp].pt.y/(float)height;
        ++it;

        //The Patch MEAN RGB and after MEAN HSV values (6 val)
        for(int i = 0; i<2; ++i){
            for (int j = 0; j<3;++j){
                if (i==0){
                    _project_descriptor.at<float>(kp,it)=avgPixelIntensity_rgb.val[j];


                }else if (i==1){
                    if(j==2) continue; //leave intensity from the descriptor
                    if(j==0){
                        //cosine & sine HUE
                        double h = avgPixelIntensity_hsv.val[j];
                        double cos_h = std::cos(h);
                        double sin_h = (double) std::fabs(std::sin(h));
                        _project_descriptor.at<float>(kp,it)=cos_h;
                        ++it;
                        _project_descriptor.at<float>(kp,it)=sin_h;
                    }else{
                        _project_descriptor.at<float>(kp,it)=avgPixelIntensity_hsv.val[j];
                    }

                }
                ++it;
            }
        }

        //Image moments (24 val)
        for (int i = 0;i<2;++i){
            for(int j = 0; j<3;++j){
                cv::Moments mom;
                cv::Mat res_roi;
                if(i==0){
                    res_roi = rgb_channels[j]( roi );
                }if(i==1){
                    res_roi = hsv_channels[j]( roi );
                }
                mom = cv::moments(res_roi);
                _project_descriptor.at<float>(kp,it)=mom.nu12;
                ++it;
                _project_descriptor.at<float>(kp,it)=mom.nu21;
                ++it;
                _project_descriptor.at<float>(kp,it)=mom.nu30;
                ++it;
                _project_descriptor.at<float>(kp,it)=mom.nu03;
                ++it;

            }

        }

        //Eigen values
        //        for (int ev=0; ev<6; ++ev){
        //            _project_descriptor.at<float>(kp,it)=desc_imgs.eigen_vals.at<Vec6f>(keypoint[kp].pt.x, keypoint[kp].pt.y)[ev];
        //            ++it;
        //        }

        //Distance transform values (81 val)
        for (int i=0; i<roiWidth; ++i){
            for (int j=0; j<roiHeight; ++j){
                _project_descriptor.at<float>(kp,it)=image_roi_dist_transf.at<float>(i,j);
                ++it;
            }
        }
        //            std::cout<<"ITERATIONS: "<<it<<std::endl;
    }


    descriptors=_project_descriptor;
    if (seeSize)
        std::cout<<"PROJECT DECRIPTOR SIZE: WIDTH: "<<_project_descriptor.cols<<" HEIGHT: "<<_project_descriptor.rows<<std::endl;


}

void Descriptor::doctorateDescriptor(cv::Mat const& _img, std::vector<cv::KeyPoint> const&keypoint, uint _patch_size,bool seeSize){


    std::vector<int>patchMeanActive,second_cm_act,third_cm_act,fourth_cm_act,fifth_cm_act,hu_mom_act,gev_l_act,gev_c_act,goesb_act,ami_act,dt_act,eigen_act,grad_act;

    patchMeanActive = {1,0,1,1,1,0}; // 4
    second_cm_act = {0,1,1,0,0,0,1,1,0}; //4
    third_cm_act={1,0,0,1,0,1,0,0,0,1,0,1}; // 5
    fourth_cm_act={0,0,1,0,1,0,0,0,0,1,0,1,1,1,0}; // 6
    fifth_cm_act={1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0}; // 10
    hu_mom_act={0,1,0,0,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,1,0}; // 13
    gev_l_act={1,0,1,1,0,1}; // 4
    gev_c_act={1,0,0,1,1,1}; // 4
    goesb_act={0,0,0,1}; // 1
    ami_act={0,1,0,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0}; // 11
    dt_act={1,1}; // 2
    eigen_act={1,0,0,1,1,0,1,0,0}; // 4
    grad_act={0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,0}; // 23


    vector<float> calculated_values;


    //size of the descriptor vector (2+6+24+(_patch_size*_patch_size))
    //    int N = 113;
    //    int N = 2+6+24+(_patch_size*_patch_size);
    int N = 93;
    uint radius = (_patch_size-1)/2;
    int width = _img.cols;
    int height = _img.rows;
    Mat _project_descriptor=Mat::zeros(keypoint.size(),N,CV_32FC1);




    //! Prepare image for descriptor extraction --> bluring, edge extraction, distance transform creation
    DescriptorImages desc_imgs;
    desc_imgs.prepareImageForTestDescriptor(_img);



    //!=====================================================

    vector<Mat> hsv_channels(3);
    vector<Mat> rgb_channels(3);
    vector<Mat> converted_im_channnels(3);
    vector<Mat> float_im_channels(3);

    cv::split(desc_imgs.blured_rgb_img,rgb_channels);
    cv::split(desc_imgs.blured_hsv_img,hsv_channels);
    cv::split(desc_imgs.converted_img,converted_im_channnels);
    cv::split(desc_imgs.float_img,float_im_channels);




    //calculate at the keypoints the patches and descriptor values
    for (uint kp=0; kp<keypoint.size();++kp){
        int roiWidth = radius*2+1;
        int roiHeight = radius*2+1;
        int roiVertexXCoordinate = keypoint[kp].pt.x - radius;
        int roiVertexYCoordinate = keypoint[kp].pt.y - radius;


        //defines roi
        cv::Rect roi( roiVertexXCoordinate, roiVertexYCoordinate, roiWidth, roiHeight );

        //copies input image in roi
        cv::Mat image_roi_rgb = desc_imgs.blured_rgb_img( roi );
        cv::Mat image_roi_hsv = desc_imgs.blured_hsv_img( roi );
        cv::Mat image_roi_conv = desc_imgs.converted_img( roi );
        cv::Mat image_roi_dist_transf = desc_imgs.dist_trans_img( roi );
        cv::Mat gevers_3l_mat = desc_imgs.gevers3l( roi );
        cv::Mat gevers_3c_mat = desc_imgs.gevers3c( roi );
        cv::Mat geusebroek_HC_mat = desc_imgs.geusebroekHC( roi );



        //FEED the descriptor with data
        int it =0;

        //The X an Y position (2 val)
        _project_descriptor.at<float>(kp,it)= (float)keypoint[kp].pt.x/(float)width;
        ++it;
        _project_descriptor.at<float>(kp,it)= (float)keypoint[kp].pt.y/(float)height;
        ++it;


        //!===================================================================================
        //!The Patch MEAN and Std (6 val)

        Mat mean,stdDev;
        cv::meanStdDev(image_roi_conv,mean,stdDev);

        for(int ms_it=0;ms_it<2;++ms_it){
            for(int val_it=0;val_it<3;++val_it){

                std::vector<float>_res;
                if(ms_it==0){
                    mean.copyTo(_res);
                }else if(ms_it==1){
                    stdDev.copyTo(_res);
                }
                calculated_values.push_back(_res[val_it]);
            }
        }

        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<patchMeanActive.size();++ae_i){

            if(patchMeanActive.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The Central Moments

        std::vector<Moments> mom_im_vec;
        std::vector<CentralMoments> high_mom_im_vec;
        std::vector<HuMomentInvariant> hu_mom_vec;
        std::vector<AffineMoments> ami_vec;

        for(int m_j = 0; m_j<3;++m_j){
            cv::Moments mom;
            cv::Mat res_patch;
            res_patch = converted_im_channnels[m_j]( roi );

            mom = cv::moments(res_patch);
            mom_im_vec.push_back(mom);

            CentralMoments c_moms = CentralMoments(mom);
            MomentsCounter::calculateHighCentralMoment(res_patch,mom.m00,mom.m01,mom.m10, c_moms);
            high_mom_im_vec.push_back(c_moms);

            HuMomentInvariant hu_mom(mom);
            hu_mom_vec.push_back(hu_mom);

            AffineMoments ami = AffineMoments(c_moms);
            ami_vec.push_back(ami);

            res_patch.release();
        }

        //!===================================================================================
        //!The 2nd central moments (9 val)


        for(int s_j = 0; s_j<3;++s_j){

            calculated_values.push_back((float) mom_im_vec[s_j].nu20);
            calculated_values.push_back((float) mom_im_vec[s_j].nu11);
            calculated_values.push_back((float) mom_im_vec[s_j].nu02);

        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<second_cm_act.size();++ae_i){

            if(second_cm_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The 3rd central moments (12 val)


        for(int s_j = 0; s_j<3;++s_j){

            calculated_values.push_back((float) mom_im_vec[s_j].nu30);
            calculated_values.push_back((float) mom_im_vec[s_j].nu21);
            calculated_values.push_back((float) mom_im_vec[s_j].nu12);
            calculated_values.push_back((float) mom_im_vec[s_j].nu03);

        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<third_cm_act.size();++ae_i){

            if(third_cm_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The 4th central moments (15 val)


        for(int s_j = 0; s_j<3;++s_j){

            calculated_values.push_back((float) high_mom_im_vec[s_j].nu40);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu31);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu22);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu13);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu04);
        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<fourth_cm_act.size();++ae_i){

            if(fourth_cm_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The 5th central moments (18 val)


        for(int s_j = 0; s_j<3;++s_j){


            calculated_values.push_back((float) high_mom_im_vec[s_j].nu50);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu41);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu32);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu23);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu14);
            calculated_values.push_back((float) high_mom_im_vec[s_j].nu05);

        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<fifth_cm_act.size();++ae_i){

            if(fifth_cm_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();


        //!===================================================================================
        //!The Hu moments (24 val)


        for(int s_j = 0; s_j<3;++s_j){

            calculated_values.push_back((float) hu_mom_vec[s_j].hu_01);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_02);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_03);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_04);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_05);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_06);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_07);
            calculated_values.push_back((float) hu_mom_vec[s_j].hu_08);

        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<hu_mom_act.size();++ae_i){

            if(hu_mom_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();



        //!===================================================================================
        //!The Gevers l moments (6 val)


        cv::meanStdDev(gevers_3l_mat,mean,stdDev);

        for(int ms_it=0;ms_it<2;++ms_it){
            for(int val_it=0;val_it<3;++val_it){

                std::vector<float>_res;
                if(ms_it==0){
                    mean.copyTo(_res);
                }else if(ms_it==1){
                    stdDev.copyTo(_res);
                }
                calculated_values.push_back(_res[val_it]);
            }
        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<gev_l_act.size();++ae_i){

            if(gev_l_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The Gevers c moments (6 val)


        cv::meanStdDev(gevers_3c_mat,mean,stdDev);

        for(int ms_it=0;ms_it<2;++ms_it){
            for(int val_it=0;val_it<3;++val_it){

                std::vector<float>_res;
                if(ms_it==0){
                    mean.copyTo(_res);
                }else if(ms_it==1){
                    stdDev.copyTo(_res);
                }
                calculated_values.push_back(_res[val_it]);
            }
        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<gev_c_act.size();++ae_i){

            if(gev_c_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The Goesbroek (4 val)


        cv::meanStdDev(geusebroek_HC_mat,mean,stdDev);

        for(int ms_it=0;ms_it<2;++ms_it){
            for(int val_it=0;val_it<3;++val_it){

                std::vector<float>_res;
                if(ms_it==0){
                    mean.copyTo(_res);
                }else if(ms_it==1){
                    stdDev.copyTo(_res);
                }
                calculated_values.push_back(_res[val_it]);
            }
        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<goesb_act.size();++ae_i){

            if(goesb_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The AMI moments (24 val)


        for(int s_j = 0; s_j<3;++s_j){

            calculated_values.push_back((float) ami_vec[s_j].ami_01);
            calculated_values.push_back((float) ami_vec[s_j].ami_02);
            calculated_values.push_back((float) ami_vec[s_j].ami_03);
            calculated_values.push_back((float) ami_vec[s_j].ami_04);
            calculated_values.push_back((float) ami_vec[s_j].ami_05);
            calculated_values.push_back((float) ami_vec[s_j].ami_06);
            calculated_values.push_back((float) ami_vec[s_j].ami_07);
            calculated_values.push_back((float) ami_vec[s_j].ami_08);
            calculated_values.push_back((float) ami_vec[s_j].ami_09);
            calculated_values.push_back((float) ami_vec[s_j].ami_10);

        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<ami_act.size();++ae_i){

            if(ami_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();


        //!===================================================================================
        //!The Distance Transfer (2 val)


        cv::meanStdDev(image_roi_dist_transf,mean,stdDev);

        calculated_values.push_back((float) mean.at<double>(0,0));
        calculated_values.push_back((float) stdDev.at<double>(0,0));


        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<dt_act.size();++ae_i){

            if(dt_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();

        //!===================================================================================
        //!The Eigen (9 val)


        for(int eig_it=0;eig_it<3;++eig_it){

            Mat eigenValues,eigenVectors;
            cv::eigen(float_im_channels[eig_it]( roi ),eigenValues,eigenVectors);


            //Loads to descriptor the first three eigen values
            for(int e_val_it=0;e_val_it<3;++e_val_it){
                calculated_values.push_back(eigenValues.at<float>(e_val_it,0));
            }

            eigenValues.release();
            eigenVectors.release();
        }



        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<eigen_act.size();++ae_i){

            if(eigen_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();


        //!===================================================================================
        //!Image Gradients (45 val)


        //        int radius =_patch.radius();
        vector<vector<float>> average_angles_all;


        //defines roi
        vector<cv::Rect>regions(4);

        regions[0] = cv::Rect( 0, 0, radius, radius );
        regions[1] = cv::Rect( radius+1, 0, radius, radius );
        regions[2] = cv::Rect( 0, radius+1, radius, radius );
        regions[3] = cv::Rect( radius+1, radius+1, radius, radius );

        for (int an_it=0;an_it<3;++an_it){

            cv::Mat angle_patch = desc_imgs.gradient_angles[an_it](roi);
            vector<float> curr_average_angs;
            Scalar p_av = cv::mean(desc_imgs.gradient_angles[0](roi));    //====> check this with value 0 also
            curr_average_angs.push_back(p_av.val[0]);
            calculated_values.push_back(p_av.val[0]/360.0f);

            for(int pw_it=0;pw_it<4;++pw_it){
                Mat window =angle_patch(regions[pw_it]);
                Scalar w_av = cv::mean(window);
                curr_average_angs.push_back(w_av.val[0]);
                calculated_values.push_back(w_av.val[0]/360.0f);
                window.release();
            }

            average_angles_all.push_back(curr_average_angs);
        }


        for (int dif_it=0;dif_it<3;++dif_it){

            vector<float> curr_diff = Matek::calculateAngleDifferences(average_angles_all[0]);

            for(int vl_it=0;vl_it<(int) curr_diff.size();++vl_it){

                calculated_values.push_back(curr_diff[vl_it]/360.0f);
            }

        }


        //!Returns only the active values of the module

        for(uint ae_i=0;ae_i<grad_act.size();++ae_i){

            if(grad_act.at(ae_i)==1){
                _project_descriptor.at<float>(kp,it) = calculated_values.at(ae_i);
                ++it;
            }else{
                continue;
            }
        }
        calculated_values.clear();


    }


    descriptors=_project_descriptor;
    if (seeSize)
        std::cout<<"PROJECT DECRIPTOR SIZE: WIDTH: "<<_project_descriptor.cols<<" HEIGHT: "<<_project_descriptor.rows<<std::endl;

    cv::waitKey();


}

void Descriptor::isFloatMatrix(cv::Mat &rawMat){


    if(rawMat.depth()!=CV_32F){
        rawMat.convertTo(descriptors,CV_32FC1);
    }else{
        descriptors = rawMat;
    }

}


