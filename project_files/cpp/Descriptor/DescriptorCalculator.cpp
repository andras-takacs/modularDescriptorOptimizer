#include "Descriptor/DescriptorCalculator.h"

DescriptorCalculator::DescriptorCalculator()
{

}

DescriptorCalculator::~DescriptorCalculator()
{

}

DescriptorCalculator::DescriptorCalculator(std::shared_ptr<MDADescriptor>_descriptor)
{
    descriptorModuleVector = _descriptor->getModuleList();
    calculatedDescriptor = _descriptor;

//        std::cout<<"Module size before calculate: "<<descriptorModuleVector.size()<<std::endl;
//        std::cout<<"Module members: "<<std::endl;
//        for(int _it=0;_it<descriptorModuleVector.size();++_it){
//        std::cout<<"Module "<<descriptorModuleVector[_it].name()<<std::endl;
//        }

}

Mat DescriptorCalculator::calculate(Images& _in_im, vector<KeyPoint> const& _keypoint){

    Mat result_image_descriptor;



    int radius = calculatedDescriptor->getPatchRadius();

//    std::cout<<"Patch radius: "<<radius<<std::endl;

    // Prepare image for descriptor extraction --> bluring, edge extraction, distance transform creation
    DescriptorImages desc_imgs;
    desc_imgs.prepareImageForModularDescriptor(_in_im.col_im,calculatedDescriptor);

    //!=====================================================

    vector<Mat> hsv_channels(3);
    vector<Mat> rgb_channels(3);
    vector<Mat> converted_im_channnels(3);
    vector<Mat> float_im_channels(3);

    cv::split(desc_imgs.blured_rgb_img,rgb_channels);
    cv::split(desc_imgs.blured_hsv_img,hsv_channels);
    cv::split(desc_imgs.converted_img,converted_im_channnels);
    cv::split(desc_imgs.float_img,float_im_channels);


    int roiWidth = radius*2+1;
    int roiHeight = radius*2+1;

    //calculate at the keypoints the patches and descriptor values
    for (uint kp=0; kp<_keypoint.size();++kp){
        ImagePatch im_patch(roiWidth,roiHeight,radius);

        im_patch.centerPoint = _keypoint.at(kp);
        im_patch.setFullImageHeight(desc_imgs.blured_rgb_img.rows);
        im_patch.setFullImageWidth(desc_imgs.blured_rgb_img.cols);

        int roiVertexXCoordinate = _keypoint[kp].pt.x - radius;
        int roiVertexYCoordinate = _keypoint[kp].pt.y - radius;

        //        if(kp==0)
        //        std::cout<<"First Keypoint: "<<keypoint[kp].pt.x<<", "<<keypoint[kp].pt.y<<std::endl;

        //defines roi
        cv::Rect roi( roiVertexXCoordinate, roiVertexYCoordinate, roiWidth, roiHeight );

        //copies input image in roi
        im_patch.rgb_image = desc_imgs.blured_rgb_img( roi );
        im_patch.hsv_image = desc_imgs.blured_hsv_img( roi );
//        im_patch.rgb_image = ImageProcessing::normalizeWithLocalMax(im_patch.rgb_image);
//        im_patch.hsv_image = ImageProcessing::normalizeWithLocalMax(im_patch.hsv_image);

        im_patch.converted_imge = desc_imgs.converted_img( roi );
        im_patch.gevers_3l_mat = desc_imgs.gevers3l( roi );
        im_patch.gevers_3c_mat = desc_imgs.gevers3c( roi );
        im_patch.geusebroek_HC_mat = desc_imgs.geusebroekHC( roi );
        im_patch.distance_transfer = desc_imgs.dist_trans_img( roi );


        for(int j=0;j<3;++j){
            im_patch.rgb_channels.push_back(rgb_channels[j]( roi ));
            im_patch.hsv_channels.push_back(hsv_channels[j]( roi ));
            im_patch.converted_channels.push_back(converted_im_channnels[j]( roi ));
            im_patch.float_im_channels.push_back(float_im_channels[j]( roi ));
            im_patch.angle_channels.push_back(desc_imgs.gradient_angles[j](roi));
        }



//        Mat _res(1,1,CV_32FC1,cv::Scalar::all(0));
        Mat _res;
        _res = calculateOnePatch(im_patch);



//        _keypoint[kp].angle = im_patch.centerPoint.angle;

        result_image_descriptor.push_back(_res);


    }



    return result_image_descriptor;
}

Mat DescriptorCalculator::calculateOnePatch(ImagePatch& _in_patch)
{

    vector<float> descriptor_result_vector;

//    std::cout<<"Module vector size: "<<descriptor_result_vector.size()<<std::endl;

    for(uint it=0;it<descriptorModuleVector.size();++it){
        vector<float>current_descriptor_results;
        DescriptorModule curr_mod = descriptorModuleVector[it];

        curr_mod.calculate(_in_patch);

        current_descriptor_results = curr_mod.getValues();


        if(it == 0){
            descriptor_result_vector = current_descriptor_results;
        }else{
            concatVectors(descriptor_result_vector,current_descriptor_results);
        }
    }


    Mat descriptorResults(Mat::zeros(1,(int) descriptor_result_vector.size(),CV_32FC1));
    descriptorResults = vectorToMat(descriptor_result_vector);

    return descriptorResults;
}

void DescriptorCalculator::concatVectors(vector<float>& _base_vector, vector<float>& _added_vector){

    for(uint v_it=0;v_it<_added_vector.size();++v_it){

        _base_vector.push_back(_added_vector[v_it]);
    }

}

Mat DescriptorCalculator::vectorToMat(vector<float>& _write_out_vector){

    Mat result_matrix(Mat::zeros(1,(int) _write_out_vector.size(),CV_32FC1));

    for(uint wo_it=0;wo_it<_write_out_vector.size();++wo_it){

        result_matrix.at<float>(0,wo_it) = (float) _write_out_vector[wo_it];

    }

    return result_matrix;

}
