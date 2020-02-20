#include "Descriptor/DescriptorModule.h"

DescriptorModule::DescriptorModule()
{
    module_id = 0;
    module_size = 0;
    default_module_size =0;
    module_position = 0;
    calculation_time = 0.0;
    module_name = "Not set";
    calculated = false;
    isActive = false;
}

DescriptorModule::DescriptorModule(int& _module_id, vector<int>& _active_elements, int _size, int _module_position, int _project_fase)
{
    module_id = _module_id;
    active_values = _active_elements;
    module_position = _module_position;
    calculation_time = 0.0;
    calculated = false;
    isActive = true;
    module_name = setModuleName(_module_id);
    default_module_size = setDefaultModuleSize(_module_id);
    module_size = _size;
    projectFase = _project_fase;

}

DescriptorModule::~DescriptorModule()
{

}


void DescriptorModule::calculate(ImagePatch& _patch){


    //    Mat _module_descriptor=Mat::zeros(0,0,CV_32FC1);

    if(projectFase==BASE){

        if(module_id==POSITION){

            //        _module_descriptor = loadPosition(_patch);
            loadPosition(_patch);


        }else if(module_id==AV_COLOR_VALUES){

            //        _module_descriptor = calculateAverageColorValues(_patch);
            calculateAverageColorValues(_patch);

        }else if(module_id==CENTRAL_MOMENTS){

            //        _module_descriptor = calculateCentralMomentValues(_patch);
            calculateCentralMomentValues(_patch);

        }else if(module_id==DISTANCE_TRANSFORM){

            //        _module_descriptor = loadDistanceTransformValues(_patch);
            loadDistanceTransformValues(_patch);
        }


        //    descriptor_values = _module_descriptor;
        isActive = true;
        //    _module_descriptor.release();
        //    return descriptor_values;
    }else{

        if(module_id==PATCH_POSITION){

            loadPosition(_patch);

        }else if(module_id==COLOR_MEAN_STD){

            calculateMeanAndStdDevForPatch(_patch);

        }else if(module_id==CENTRAL_MOMENTS_2ND){

            calculate2ndCentralMoment(_patch);

        }else if(module_id==CENTRAL_MOMENTS_3RD){

            calculate3rdCentralMoment(_patch);

        }else if(module_id==CENTRAL_MOMENTS_4TH){

            calculate4thCentralMoment(_patch);

        }else if(module_id==CENTRAL_MOMENTS_5TH){

            calculate5thCentralMoment(_patch);

        }else if(module_id==HU_MOMENTS){

            calculateHuMoments(_patch);

        }else if(module_id==AFFINE_MOMENTS){

            calculateAffineMoments(_patch);

        }else if(module_id==GEVERS_3L){

            calculateGevers3L(_patch);

        }else if(module_id==GEVERS_3C){

            calculateGevers3C(_patch);

        }else if(module_id==GEUSEBROEK_H_C){

            calculateGeusebroekHC(_patch);

        }else if(module_id==EIGEN_3){

            calculateEigenValues(_patch);

        }else if(module_id==DT_MSTD){

            calculateDTMeanAndSTD(_patch);

        }else if(module_id==GRADIENT_CAL){

            calculateGradients(_patch);
        }
        isActive = true;
    }
}

string DescriptorModule::setModuleName(int &_module_id){

    string _name ="";

    if(projectFase==BASE){

        switch (_module_id)
        {
        case POSITION:
            _name = "Position";
            break;
        case AV_COLOR_VALUES:
            _name = "Average Color Values";
            break;
        case CENTRAL_MOMENTS:
            _name = "Central Moments";
            break;
        case DISTANCE_TRANSFORM:
            _name="Distance Transform";
            break;
        default:
            _name = "Not Set!!";
            break;
        }

    }else{

        switch (_module_id)
        {
        case PATCH_POSITION:
            _name = "Position coordinates";
            break;
        case CENTRAL_MOMENTS_2ND:
            _name = "2nd Order Normalized Central Moments (nu20, nu11, nu02)";
            break;
        case CENTRAL_MOMENTS_3RD:
            _name = "3rd Order Normalized Central Moments (nu30, nu21, nu12, nu03)";
            break;
        case CENTRAL_MOMENTS_4TH:
            _name = "4th Order Normalized Central Moments (nu40, nu31, nu22, nu13, nu04)";
            break;
        case CENTRAL_MOMENTS_5TH:
            _name = "5th Order Normalized Central Moments (nu50, nu41, nu32, nu23, nu14, nu05)";
            break;
        case HU_MOMENTS:
            _name = "Hu Moments (hu1, hu2, hu3, hu4, hu5, hu6, hu7, hu8)";
            break;
        case AFFINE_MOMENTS:
            _name = "Suk and Flusser Affine Moment Invariant (ami1, ami2, ami3, ami4, ami5, ami6, ami7, ami8, ami9, ami10)";
            break;
        case COLOR_MEAN_STD:
            _name = "The mean and standard deviation of each channels";
            break;
        case GEVERS_3L:
            _name="Gevers l1,l2,l3 invariant features (mean, standard deviation)";
            break;
        case GEVERS_3C:
            _name="Gevers c1,c2,c3 invariant features (mean, standard deviation)";
            break;
        case GEUSEBROEK_H_C:
            _name="Geusebroek H and C invariant features (mean, standard deviation)";
            break;
        case DT_MSTD:
            _name="Distnace Transform (mean, standard deviation)";
            break;
        case EIGEN_3:
            _name="Eigen Values (The three highest)";
            break;
        case GRADIENT_CAL:
            _name="Image Gradient angles (averages and differences)";
            break;
        default:
            _name = "Not Set!!";
            break;
        }
    }

    return _name;
}

int DescriptorModule::setDefaultModuleSize(int &_module_id){

    int _size =0;

    if(projectFase==BASE){

        switch (_module_id)
        {
        case POSITION:
            _size = 2;
            break;
        case AV_COLOR_VALUES:
            _size = 6;
            break;
        case CENTRAL_MOMENTS:
            _size = 24;
            break;
        case DISTANCE_TRANSFORM:
            _size = 81;
            break;
        default:
            _size = 0;
            break;
        }
    }else{
        switch (_module_id)
        {
        case PATCH_POSITION:
            _size = 2;
            break;
        case CENTRAL_MOMENTS_2ND:
            _size = 9;
            break;
        case CENTRAL_MOMENTS_3RD:
            _size = 12;
            break;
        case CENTRAL_MOMENTS_4TH:
            _size = 15;
            break;
        case CENTRAL_MOMENTS_5TH:
            _size = 18;
            break;
        case HU_MOMENTS:
            _size = 24;
            break;
        case AFFINE_MOMENTS:
            _size = 30;
            break;
        case COLOR_MEAN_STD:
            _size = 6;
            break;
        case GEVERS_3L:
            _size = 6;
            break;
        case GEVERS_3C:
            _size = 6;
            break;
        case GEUSEBROEK_H_C:
            _size = 4;
            break;
        case DT_MSTD:
            _size = 2;
            break;
        case EIGEN_3:
            _size = 9;
            break;
        case GRADIENT_CAL:
            _size = 45;
            break;
        default:
            _size = 0;
            break;
        }

    }

    return _size;
}

string DescriptorModule::writeOutModuleGenome(){

    stringstream ss_genome;
    std::copy( active_values.begin(), active_values.end(), ostream_iterator<int>(ss_genome, ""));
    string string_out = ss_genome.str();
    string_out = string_out.substr(0, string_out.length());

    return string_out;
}

void DescriptorModule::loadPosition(ImagePatch& _patch){

    //calculation of position module
    std::clock_t module_calculation_time = std::clock();

    //    Mat _position_descriptor;
    //    Mat _position_descriptor=Mat::zeros(0,0,CV_32FC1);
    //! CALCULATION OF MODULE POSITION -->
    //! saving the patch (x,y) position (2 val)

    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            if(ae_i==0){
                module_values.push_back((float)_patch.centerPoint.pt.x/(float)_patch.img_width());

            }else if (ae_i==1){
                module_values.push_back((float)_patch.centerPoint.pt.y/(float)_patch.img_height());
            }
        }else{
            continue;
        }

    }
    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    //! The push back creates a column matrix... Needs to be transposed
    //    cv::transpose(_position_descriptor,_position_descriptor);

    //    return _position_descriptor;

}

void DescriptorModule::calculateAverageColorValues(ImagePatch& _patch){


    //    Mat _average_color_descriptor=Mat::zeros(0,0,CV_32FC1);
    //    Mat _average_color_descriptor;
    vector<float> calculated_values;

    //calculation of position module
    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Average Colour Values -->
    //! MEAN RGB and after MEAN HSV values (6 val)

    cv::Scalar avgPixelIntensity_rgb = cv::mean( _patch.rgb_image );
    cv::Scalar avgPixelIntensity_hsv = cv::mean( _patch.hsv_image );


    //The Patch MEAN RGB and after MEAN HSV values (6 val)
    for(int i = 0; i<2; ++i){
        for (int j = 0; j<3;++j){
            if (i==0){
                calculated_values.push_back((float) avgPixelIntensity_rgb.val[j]);

            }else if (i==1){
                if(j==2) continue; //leave intensity from the descriptor
                if(j==0){
                    //cosine & sine HUE
                    float h = 0.0, cos_h = 0.0, sin_h = 0.0;
                    h = (float) avgPixelIntensity_hsv.val[j];
                    cos_h = std::cos(h);
                    sin_h = std::fabs(std::sin(h));
                    calculated_values.push_back(cos_h);
                    calculated_values.push_back(sin_h);
                }else{
                    calculated_values.push_back((float) avgPixelIntensity_hsv.val[j]);
                }

            }
        }
    }

    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            //            _average_color_descriptor.push_back(calculated_values.at(ae_i));
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    //! The push back creates a column matrix... Needs to be transposed
    //    cv::transpose(_average_color_descriptor,_average_color_descriptor);

    //    return _average_color_descriptor;

}



void DescriptorModule::calculateCentralMomentValues(ImagePatch& _patch){

    //    Mat _central_moment_descriptor;
    //    Mat _central_moment_descriptor=Mat::zeros(0,0,CV_32FC1);
    vector<float> calculated_values;

    //calculation of position module
    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 3rd degree central moments for each channel)(24 val)

    for (int i = 0;i<2;++i){
        for(int j = 0; j<3;++j){
            cv::Moments mom;
            cv::Mat res_patch;
            if(i==0){
                res_patch = _patch.rgb_channels[j];
            }if(i==1){
                res_patch = _patch.hsv_channels[j];
            }
            mom = cv::moments(res_patch);
            calculated_values.push_back((float) mom.nu12);
            calculated_values.push_back((float) mom.nu21);
            calculated_values.push_back((float) mom.nu30);
            calculated_values.push_back((float) mom.nu03);

        }

    }

    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            //            _central_moment_descriptor.push_back(calculated_values.at(ae_i));
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    //! The push back creates a column matrix... Needs to be transposed
    //    cv::transpose(_central_moment_descriptor,_central_moment_descriptor);

    //    return _central_moment_descriptor;

}

void DescriptorModule::loadDistanceTransformValues(ImagePatch& _patch){


    vector<float> calculated_values;

    //calculation of position module
    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Distance transform values -->
    //! White (large value) further away from the edge

    for (int i=0; i<_patch.width(); ++i){
        for (int j=0; j<_patch.height(); ++j){
            calculated_values.push_back(_patch.distance_transfer.at<float>(i,j));
        }
    }

    //std::cout<<"Active value size: "<<calculated_values.size()<<std::endl;
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            //            _distance_transform_descriptor.push_back(calculated_values.at(ae_i));
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    //! The push back creates a column matrix... Needs to be transposed
    //    cv::transpose(_distance_transform_descriptor,_distance_transform_descriptor);

    //    return _distance_transform_descriptor;

}

void DescriptorModule::calculateMeanAndStdDevForPatch(ImagePatch& _patch){

    vector<float> calculated_values;
    Mat mean,stdDev;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Mean and Standard Deviation (6 val) ==>
    //! OpenCV function returns back two column matrix with the results of
    //! the patch means for each channel and Standard deviation for each channel
    //! =========================================================================
    //! The returnes values are:
    //! 1 - mean(CH1),
    //! 2 - mean(CH2),
    //! 3 - mean(CH3),
    //! 4 - StD(CH1),
    //! 5 - StD(CH2),
    //! 6 - StD(CH3);
    //! =========================================================================
    cv::meanStdDev(_patch.converted_imge,mean,stdDev);

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
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    mean.release();
    stdDev.release();

}

void DescriptorModule::calculate2ndCentralMoment(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 2nd degree central moments for each channel)(9 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - nu20(CH1),
    //! 2 - nu11(CH1),
    //! 3 - nu02(CH1),
    //! 4 - nu20(CH2),
    //! 5 - nu11(CH2),
    //! 6 - nu02(CH2),
    //! 7 - nu20(CH3),
    //! 8 - nu11(CH3),
    //! 9 - nu02(CH3);
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;

        res_patch = _patch.converted_channels[j];

        mom = cv::moments(res_patch);
        calculated_values.push_back((float) mom.nu20);
        calculated_values.push_back((float) mom.nu11);
        calculated_values.push_back((float) mom.nu02);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;
}

void DescriptorModule::calculate3rdCentralMoment(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 3nd degree central moments for each channel)(12 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - nu30(CH1),
    //! 2 - nu21(CH1),
    //! 3 - nu12(CH1),
    //! 4 - nu03(CH1),
    //! 5 - nu30(CH2),
    //! 6 - nu21(CH2),
    //! 7 - nu12(CH2),
    //! 8 - nu03(CH2),
    //! 9 - nu30(CH3);
    //! 10- nu21(CH3);
    //! 11- nu12(CH3);
    //! 12- nu03(CH3);
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;

        res_patch = _patch.converted_channels[j];

        mom = cv::moments(res_patch);
        calculated_values.push_back((float) mom.nu30);
        calculated_values.push_back((float) mom.nu21);
        calculated_values.push_back((float) mom.nu12);
        calculated_values.push_back((float) mom.nu03);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

}

void DescriptorModule::calculate4thCentralMoment(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 4th degree central moments for each channel)(15 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - nu40(CH1),
    //! 2 - nu31(CH1),
    //! 3 - nu22(CH1),
    //! 4 - nu13(CH1),
    //! 5 - nu04(CH1),
    //! 6 - nu40(CH2),
    //! 7 - nu31(CH2),
    //! 8 - nu22(CH2),
    //! 9 - nu13(CH2);
    //! 10- nu04(CH2);
    //! 11- nu40(CH3);
    //! 12- nu31(CH3);
    //! 13- nu22(CH3);
    //! 14- nu13(CH3);
    //! 15- nu04(CH3);
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;
        res_patch = _patch.converted_channels[j];
        mom = cv::moments(res_patch);

        CentralMoments c_moms = CentralMoments(mom);
        MomentsCounter::calculateHighCentralMoment(res_patch,mom.m00,mom.m01,mom.m10, c_moms);

        calculated_values.push_back((float) c_moms.nu40);
        calculated_values.push_back((float) c_moms.nu31);
        calculated_values.push_back((float) c_moms.nu22);
        calculated_values.push_back((float) c_moms.nu13);
        calculated_values.push_back((float) c_moms.nu04);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

}

void DescriptorModule::calculate5thCentralMoment(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 5th degree central moments for each channel)(18 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - nu50(CH1),
    //! 2 - nu41(CH1),
    //! 3 - nu32(CH1),
    //! 4 - nu23(CH1),
    //! 5 - nu14(CH1),
    //! 6 - nu05(CH1),
    //! 7 - nu50(CH2),
    //! 8 - nu41(CH2),
    //! 9 - nu32(CH2);
    //! 10- nu23(CH2);
    //! 11- nu14(CH2);
    //! 12- nu05(CH2);
    //! 13- nu50(CH3);
    //! 14- nu41(CH3);
    //! 15- nu32(CH3);
    //! 16- nu23(CH3);
    //! 17- nu14(CH3);
    //! 18- nu05(CH3);
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;
        res_patch = _patch.converted_channels[j];
        mom = cv::moments(res_patch);

        CentralMoments c_moms = CentralMoments(mom);
        MomentsCounter::calculateHighCentralMoment(res_patch,mom.m00,mom.m01,mom.m10, c_moms);

        calculated_values.push_back((float) c_moms.nu50);
        calculated_values.push_back((float) c_moms.nu41);
        calculated_values.push_back((float) c_moms.nu32);
        calculated_values.push_back((float) c_moms.nu23);
        calculated_values.push_back((float) c_moms.nu14);
        calculated_values.push_back((float) c_moms.nu05);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
//            std::cout<<"5th order values: "<<calculated_values.at(ae_i)<<"\n";
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

}

void DescriptorModule::calculateHuMoments(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Image moments (normalized 5th degree central moments for each channel)(24 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - hu1(CH1),  |  9 - hu1(CH2),  |  17- hu1(CH3),
    //! 2 - hu2(CH1),  |  10- hu2(CH2),  |  18- hu2(CH3),
    //! 3 - hu3(CH1),  |  11- hu3(CH2),  |  19- hu3(CH3),
    //! 4 - hu4(CH1),  |  12- hu4(CH2),  |  20- hu4(CH3),
    //! 5 - hu5(CH1),  |  13- hu5(CH2),  |  21- hu5(CH3),
    //! 6 - hu6(CH1),  |  14- hu6(CH2),  |  22- hu6(CH3),
    //! 7 - hu7(CH1),  |  15- hu7(CH2),  |  23- hu7(CH3),
    //! 8 - hu8(CH1),  |  16- hu8(CH2),  |  24- hu8(CH3),
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;
        res_patch = _patch.converted_channels[j];
        mom = cv::moments(res_patch);

        HuMomentInvariant hu_mom(mom);

        calculated_values.push_back((float) hu_mom.hu_01);
        calculated_values.push_back((float) hu_mom.hu_02);
        calculated_values.push_back((float) hu_mom.hu_03);
        calculated_values.push_back((float) hu_mom.hu_04);
        calculated_values.push_back((float) hu_mom.hu_05);
        calculated_values.push_back((float) hu_mom.hu_06);
        calculated_values.push_back((float) hu_mom.hu_07);
        calculated_values.push_back((float) hu_mom.hu_08);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

}

void DescriptorModule::calculateAffineMoments(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Suk and Flusser Affine Moment Invariant (30 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - ami01(CH1),  |  11- ami01(CH2),  |  21- ami01(CH3),
    //! 2 - ami02(CH1),  |  12- ami02(CH2),  |  22- ami02(CH3),
    //! 3 - ami03(CH1),  |  13- ami03(CH2),  |  23- ami03(CH3),
    //! 4 - ami04(CH1),  |  14- ami04(CH2),  |  24- ami04(CH3),
    //! 5 - ami05(CH1),  |  15- ami05(CH2),  |  25- ami05(CH3),
    //! 6 - ami06(CH1),  |  16- ami06(CH2),  |  26- ami06(CH3),
    //! 7 - ami07(CH1),  |  17- ami07(CH2),  |  27- ami07(CH3),
    //! 8 - ami08(CH1),  |  18- ami08(CH2),  |  28- ami08(CH3),
    //! 9 - ami09(CH1),  |  19- ami09(CH2),  |  29- ami09(CH3),
    //! 10- ami10(CH1),  |  20- ami10(CH2),  |  30- ami10(CH3),
    //! =========================================================================================

    for(int j = 0; j<3;++j){
        cv::Moments mom;
        cv::Mat res_patch;
        res_patch = _patch.converted_channels[j];

        mom = cv::moments(res_patch);
        CentralMoments c_moms = CentralMoments(mom);
        AffineMoments ami = AffineMoments(c_moms);

        calculated_values.push_back((float) ami.ami_01);
        calculated_values.push_back((float) ami.ami_02);
        calculated_values.push_back((float) ami.ami_03);
        calculated_values.push_back((float) ami.ami_04);
        calculated_values.push_back((float) ami.ami_05);
        calculated_values.push_back((float) ami.ami_06);
        calculated_values.push_back((float) ami.ami_07);
        calculated_values.push_back((float) ami.ami_08);
        calculated_values.push_back((float) ami.ami_09);
        calculated_values.push_back((float) ami.ami_10);

        res_patch.release();
    }



    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        //!Returns only the active values of the module
        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }

    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;


}

void DescriptorModule::calculateGevers3L(ImagePatch& _patch){

    vector<float> calculated_values;
    Mat mean,stdDev;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Mean and Standard Deviation of Gevers 3l values(6 val) ==>
    //! OpenCV function returns back two column matrix with the results of
    //! the patch means for each channel and Standard deviation for each channel
    //! =========================================================================
    //! The returnes values are:
    //! 1 - mean(GL1),
    //! 2 - mean(GL2),
    //! 3 - mean(GL3),
    //! 4 - StD(GL1),
    //! 5 - StD(GL2),
    //! 6 - StD(GL3);
    //! =========================================================================
    cv::meanStdDev(_patch.gevers_3l_mat,mean,stdDev);

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
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    mean.release();
    stdDev.release();

}

void DescriptorModule::calculateGevers3C(ImagePatch& _patch){

    vector<float> calculated_values;
    Mat mean,stdDev;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Mean and Standard Deviation of Gevers 3c values(6 val) ==>
    //! OpenCV function returns back two column matrix with the results of
    //! the patch means for each channel and Standard deviation for each channel
    //! =========================================================================
    //! The returnes values are:
    //! 1 - mean(GC1),
    //! 2 - mean(GC2),
    //! 3 - mean(GC3),
    //! 4 - StD(GC1),
    //! 5 - StD(GC2),
    //! 6 - StD(GC3);
    //! =========================================================================
    cv::meanStdDev(_patch.gevers_3c_mat,mean,stdDev);

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
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    mean.release();
    stdDev.release();

}

void DescriptorModule::calculateGeusebroekHC(ImagePatch& _patch){

    vector<float> calculated_values;
    Mat mean,stdDev;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Mean and Standard Deviation of Geusebroek H and C featrues (4 val) ==>
    //! OpenCV function returns back two column matrix with the results of
    //! the patch means for each channel and Standard deviation for each channel
    //! =========================================================================
    //! The returnes values are:
    //! 1 - mean(GHC1),
    //! 2 - mean(GHC2),
    //! 3 - StD(GHC1),
    //! 4 - StD(GHC2);
    //! =========================================================================
    cv::meanStdDev(_patch.geusebroek_HC_mat,mean,stdDev);

    for(int ms_it=0;ms_it<2;++ms_it){
        for(int val_it=0;val_it<4;++val_it){

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
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    mean.release();
    stdDev.release();
}

void DescriptorModule::calculateEigenValues(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Three highest Eigen Values of the patch for each channel (9 val) ==>
    //! OpenCV function returns back two matrices with the results of
    //! eigen values in yhe first matrix and the eigen vectors in the second
    //! =========================================================================
    //! The returnes values are:
    //! 1 - eig1(C1),
    //! 2 - eig2(C1),
    //! 3 - eig3(C1),
    //! 4 - eig1(C2),
    //! 5 - eig2(C2),
    //! 6 - eig3(C2),
    //! 7 - eig1(C3),
    //! 8 - eig2(C3),
    //! 9 - eig3(C3),
    //! =========================================================================


    for(int eig_it=0;eig_it<3;++eig_it){

        Mat eigenValues,eigenVectors;
        cv::eigen(_patch.float_im_channels[eig_it],eigenValues,eigenVectors);


        //Loads to descriptor the first three eigen values
        for(int e_val_it=0;e_val_it<3;++e_val_it){
            calculated_values.push_back(eigenValues.at<float>(e_val_it,0));
        }

        eigenValues.release();
        eigenVectors.release();
    }

    //!Returns only the active values of the module
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;
}

void DescriptorModule::calculateDTMeanAndSTD(ImagePatch& _patch){

    vector<float> calculated_values;
    Mat mean,stdDev;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Mean and Standard Deviation of Distnace Transform (2 val) ==>
    //! OpenCV function returns back two column matrix with the results of
    //! the patch means for each channel and Standard deviation for each channel
    //! =========================================================================
    //! The returnes values are:
    //! 1 - mean(DT1),
    //! 2 - StD(DT1)
    //! =========================================================================
    cv::meanStdDev(_patch.distance_transfer,mean,stdDev);

    calculated_values.push_back((float) mean.at<double>(0,0));
    calculated_values.push_back((float) stdDev.at<double>(0,0));

    //!Returns only the active values of the module
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;

    mean.release();
    stdDev.release();


}

void DescriptorModule::calculateGradients(ImagePatch& _patch){

    vector<float> calculated_values;

    std::clock_t module_calculation_time = std::clock();

    //! CALCULATION OF MODULE Central Moment values -->
    //! Suk and Flusser Affine Moment Invariant (30 val)
    //! =========================================================================================
    //! The returnes values are:
    //! 1 - gradav01(CH1),  |  11- gradav01(CH3),  |  21- gradif06(CH1),  |  31- gradif06(CH2),  |  41- gradif06(CH3),
    //! 2 - gradav02(CH1),  |  12- gradav02(CH3),  |  22- gradif07(CH1),  |  32- gradif07(CH2),  |  42- gradif07(CH3),
    //! 3 - gradav03(CH1),  |  13- gradav03(CH3),  |  23- gradif08(CH1),  |  33- gradif08(CH2),  |  43- gradif08(CH3),
    //! 4 - gradav04(CH1),  |  14- gradav04(CH3),  |  24- gradif09(CH1),  |  34- gradif09(CH2),  |  44- gradif09(CH3),
    //! 5 - gradav05(CH1),  |  15- gradav05(CH3),  |  25- gradif10(CH1),  |  35- gradif10(CH2),  |  45- gradif10(CH3),
    //! 6 - gradav01(CH2),  |  16- gradif01(CH1),  |  26- gradif01(CH2),  |  36- gradif01(CH3),
    //! 7 - gradav02(CH2),  |  17- gradif02(CH1),  |  27- gradif02(CH2),  |  37- gradif02(CH3),
    //! 8 - gradav03(CH2),  |  18- gradif03(CH1),  |  28- gradif03(CH2),  |  38- gradif03(CH3),
    //! 9 - gradav04(CH2),  |  19- gradif04(CH1),  |  29- gradif04(CH2),  |  39- gradif04(CH3),
    //! 10- gradav05(CH2),  |  20- gradif05(CH1),  |  30- gradif05(CH2),  |  40- gradif05(CH3),
    //! =========================================================================================

    int radius =_patch.radius();
    vector<Mat> angle_mats = _patch.angle_channels;
    vector<vector<float>> average_angles_all;


    //defines roi
    vector<cv::Rect>regions(4);

    regions[0] = cv::Rect( 0, 0, radius, radius );
    regions[1] = cv::Rect( radius+1, 0, radius, radius );
    regions[2] = cv::Rect( 0, radius+1, radius, radius );
    regions[3] = cv::Rect( radius+1, radius+1, radius, radius );

    for (int an_it=0;an_it<3;++an_it){

        vector<float> curr_average_angs;
        Scalar p_av = mean(angle_mats[an_it]);
        curr_average_angs.push_back(p_av.val[0]);
        calculated_values.push_back(p_av.val[0]/360.0f);

        for(int pw_it=0;pw_it<4;++pw_it){
            Mat window =angle_mats[an_it](regions[pw_it]);
            Scalar w_av = mean(window);
            curr_average_angs.push_back(w_av.val[0]);
            calculated_values.push_back(w_av.val[0]/360.0f);
            window.release();
        }

        average_angles_all.push_back(curr_average_angs);
    }


    for (int dif_it=0;dif_it<3;++dif_it){

        vector<float> curr_diff = Matek::calculateAngleDifferences(average_angles_all[0]);

        float deg_sum = std::accumulate(curr_diff.begin(), curr_diff.end(), 0.0);
        float av_deg_diff = deg_sum/(float) curr_diff.size();

        _patch.centerPoint.angle = av_deg_diff;

        for(int vl_it=0;vl_it<(int) curr_diff.size();++vl_it){

            calculated_values.push_back(curr_diff[vl_it]/360.0f);
        }

    }



    //!Returns only the active values of the module
    for(uint ae_i=0;ae_i<active_values.size();++ae_i){

        if(active_values.at(ae_i)==1){
            module_values.push_back(calculated_values.at(ae_i));
        }else{
            continue;
        }
    }

    //Save training time to class
    calculation_time = ( std::clock() - module_calculation_time ) / (double) CLOCKS_PER_SEC;
    calculated = true;
}
