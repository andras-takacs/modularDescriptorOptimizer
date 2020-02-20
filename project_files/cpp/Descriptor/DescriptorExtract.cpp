#include "Descriptor/DescriptorExtract.h"

DescriptorExtract::DescriptorExtract()
{
    descriptorType = D_PROJECT_D;
}

DescriptorExtract::~DescriptorExtract()
{

}

DescriptorExtract::DescriptorExtract(const int& _descriptorType)
{
    descriptorType = _descriptorType;
}

Mat DescriptorExtract::extraction(vector<KeyPoint>& _keypoints, Images& _im_in)
{

    cv::Mat in,mask;

    in = _im_in.col_im;
    mask = _im_in.mask_im;

    //=============================DESCRIPTOR EXTRACTOR TYPES=============================================
    Descriptor* descriptors1 = new Descriptor();

    if (descriptorType == SURF_D){

        descriptors1->descriptor_feature(_keypoints,in,SURF_D);
    }
    else if (descriptorType == SIFT_D){

        descriptors1->descriptor_feature(_keypoints,in,SIFT_D);
    }
    else if (descriptorType == OPP_SURF_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_SURF_D);
    }
    else if (descriptorType == OPP_SIFT_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_SIFT_D);
    }
    else if (descriptorType == ORB_D){

        descriptors1->descriptor_feature(_keypoints,in,ORB_D);
    }
    else if (descriptorType == BRIEF_D){

        descriptors1->descriptor_feature(_keypoints,in,BRIEF_D);
    }
    else if (descriptorType == BRISK_D){

        descriptors1->descriptor_feature(_keypoints,in,BRISK_D);
    }
    else if (descriptorType == FREAK_D){

        descriptors1->descriptor_feature(_keypoints,in,FREAK_D);
    }
    else if (descriptorType == LATCH_D){

        descriptors1->descriptor_feature(_keypoints,in,LATCH_D);
    }
    /*else if (descriptorType == OPP_ORB_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_ORB_D);
    }
    else if (descriptorType == OPP_BRIEF_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_BRIEF_D);
    }
    else if (descriptorType == OPP_BRISK_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_BRISK_D);
    }
    else if (descriptorType == OPP_FREAK_D){

        descriptors1->descriptor_feature(_keypoints,in,OPP_FREAK_D);
    }*/
    else if(descriptorType==RGB_D){
        descriptors1->descriptor_feature(_keypoints,in,RGB_D);   //Descriptor for 2radius+1 wide featurettes
    }
    else if(descriptorType==M_PROJECT_D){
        descriptors1->descriptor_feature(_keypoints,in,M_PROJECT_D);   //Descriptor for 2radius+1 wide featurettes
    }
    else if(descriptorType==D_PROJECT_D){
        descriptors1->descriptor_feature(_keypoints,in,D_PROJECT_D);   //Descriptor for 2radius+1 wide featurettes
    }


    //=================================Labeling Keypoints==================================


    in.release();

    return descriptors1->descriptors;
}
