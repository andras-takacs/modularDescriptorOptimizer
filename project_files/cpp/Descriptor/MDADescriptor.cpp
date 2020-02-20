#include "Descriptor/MDADescriptor.h"

MDADescriptor::MDADescriptor()
{

    rTreeDepth = 0;
    numberOfTrees = 0;
    splitNumber = 0;
    patchRadius = 0;
    gaussianKernelSize = 0;
    blurSigma = 0;
    usedColorChannel = 0;
    descriptorSize = 0;
    colorChannelName = RGB_CH;
    tolerance = 0;//cambio

}

MDADescriptor::MDADescriptor(vector<int> _genome)
{
    descriptor_genome = _genome;
    rTreeDepth = 0;
    numberOfTrees = 0;
    splitNumber = 0;
    patchRadius = 0;
    gaussianKernelSize = 0;
    blurSigma = 0;
    usedColorChannel = 0;
    descriptorSize = 0;
    colorChannelName = RGB_CH;
    tolerance = 0;
}

MDADescriptor::~MDADescriptor()
{

}




void MDADescriptor::setKernelFromGenome(int _k_enum){

    int _size =0;

    switch (_k_enum)
    {
    case K_1:
        _size = 1;
        break;
    case K_3:
        _size = 3;
        break;
    case K_5:
        _size = 5;
        break;
    case K_7:
        _size = 7;
        break;
    case K_9:
        _size = 9;
        break;
    case K_11:
        _size = 11;
        break;
    case K_13:
        _size = 13;
        break;
    case K_15:
        _size = 15;
        break;
    default:
        _size = 1;
        break;
    }

    gaussianKernelSize = _size;

    //    std::cout<<"The Gaussian kernel size: "<<_size<<std::endl;
}

void MDADescriptor::setPatchRadiusFromGenome(int _p_enum){

    int _size =0;

    switch (_p_enum)
    {

    case P_3:
        _size = 3;
        break;
    case P_4:
        _size = 4;
        break;
    case P_5:
        _size = 5;
        break;
    case P_6:
        _size = 6;
        break;
    case P_7:
        _size = 7;
        break;
    case P_8:
        _size = 8;
        break;
    case P_9:
        _size = 9;
        break;
    case P_10:
        _size = 10;
        break;
    default:
        _size = 4;
        break;
    }

    patchRadius = _size;

    //    std::cout<<"The radius size: "<<_size<<std::endl;
}

void MDADescriptor::setTreeDepthFromGenome(int _t_enum){

    int _depth =0;

    switch (_t_enum)
    {
    case RTD_5:
        _depth = 5;
        break;
    case RTD_10:
        _depth = 10;
        break;
    case RTD_15:
        _depth = 15;
        break;
    case RTD_20:
        _depth = 20;
        break;
    case RTD_25:
        _depth = 25;
        break;
    case RTD_30:
        _depth = 30;
        break;
    case RTD_35:
        _depth = 35;
        break;
    case RTD_40:
        _depth = 40;
        break;
    default:
        _depth = 25;
        break;
    }

    rTreeDepth = _depth;

    //    std::cout<<"The tree depth: "<<_depth<<std::endl;

}

void MDADescriptor::setTreeNumberFromGenome(int _tn_enum){

    int _number =0;

    switch (_tn_enum)
    {
    case NT_40:
        _number = 40;
        break;
    case NT_50:
        _number = 50;
        break;
    case NT_60:
        _number = 60;
        break;
    case NT_70:
        _number = 70;
        break;
    case NT_80:
        _number = 80;
        break;
    case NT_100:
        _number = 100;
        break;
    case NT_120:
        _number = 120;
        break;
    case NT_140:
        _number = 140;
        break;
    default:
        _number = 100;
        break;
    }

    numberOfTrees = _number;

    //    std::cout<<"The number of trees: "<<_number<<std::endl;

}

void MDADescriptor::setSigmaFromGenome(int _s_enum){

    double _sigma =0;

    switch (_s_enum)
    {
    case SIG_1:
        _sigma = 0.8;
        break;
    case SIG_2:
        _sigma = 1.1;
        break;
    case SIG_3:
        _sigma = 1.5;
        break;
    case SIG_4:
        _sigma = 3;
        break;
    case SIG_5:
        _sigma = 6;
        break;
    case SIG_6:
        _sigma = 12;
        break;
    case SIG_7:
        _sigma = 24;
        break;
    case SIG_8:
        _sigma = 30;
        break;
    default:
        _sigma = 12;
        break;
    }

    blurSigma = _sigma;

    //    std::cout<<"Sigma value: "<<_sigma<<std::endl;

}

void MDADescriptor::setSplitNumberFromGenome(int _sn_enum){

    int _split =0;

    switch (_sn_enum)
    {
    case SIG_1:
        _split = 4;
        break;
    case SIG_2:
        _split = 8;
        break;
    case SIG_3:
        _split = 12;
        break;
    case SIG_4:
        _split = 16;
        break;
    case SIG_5:
        _split = 20;
        break;
    case SIG_6:
        _split = 25;
        break;
    case SIG_7:
        _split = 30;
        break;
    case SIG_8:
        _split = 35;
        break;
    default:
        _split = 4;
        break;
    }

    splitNumber = _split;

    //    std::cout<<"Split number: "<<_split<<std::endl;

}

void MDADescriptor::setColorChannelFromGenome(int _col_enum){

    int _color_ch =0;
    string _colorName = "Not set";

    switch (_col_enum)
    {
    case RGB_CH:
        _color_ch = RGB_CH;
        _colorName = "RGB Color Space";
        break;
    case Lab_CH:
        _color_ch = Lab_CH;
        _colorName = "Lab Color Space";
        break;
    case Luv_CH:
        _color_ch = Luv_CH;
        _colorName = "Luv Color Space";
        break;
    case XYZ_CH:
        _color_ch = XYZ_CH;
        _colorName = "XYZ Color Space";
        break;
    case HSV_CH:
        _color_ch = HSV_CH;
        _colorName = "HSV Color Space";
        break;
    case HLS_CH:
        _color_ch = HLS_CH;
        _colorName = "HLS Color Space";
        break;
    case YCrCb_CH:
        _color_ch = YCrCb_CH;
        _colorName = "YCrCb Color Space";
        break;
    case OPP_CH:
        _color_ch = OPP_CH;
        _colorName = "Opponent Color Space";
        break;
    default:
        _color_ch = RGB_CH;
        _colorName = "RGB Color Space";
        break;
    }

    usedColorChannel = _color_ch;
    colorChannelName = _colorName;

    //    std::cout<<"The used color channel: "<<_colorName<<std::endl;

}

void MDADescriptor::setGradientDetector(int _grad_enum){

    int _detector =0;
    string _detectorName = "Not set";

    switch (_grad_enum)
    {
    case SOBEL:
        _detector = SOBEL;
        _detectorName = "Sobel Detector";
        break;
    case SHARR:
        _detector = SHARR;
        _detectorName = "Sharr Detector";
        break;
    default:
        _detector = SOBEL;
        _detectorName = "Sobel Detector";
        break;
    }

    sobelOrSharr = _detector;
    gradientDetector = _detectorName;

}

void MDADescriptor::setCannyKernelFromGenome(int _ck_enum){

    int _size =0;

    switch (_ck_enum)
    {
    case CK_3:
        _size = 3;
        break;
    case CK_5:
        _size = 5;
        break;
//    case CK_7:
//        _size = 7;
//        break;
//    case CK_9:
//        _size = 9;
//        break;
    default:
        _size = 3;
        break;
    }

    cannyKernelSize = _size;

    //    std::cout<<"Canny kernel size: "<<_size<<std::endl;

}

void MDADescriptor::setCannyThresholdFromGenome(int _ct_enum){

    int _thold =0;

    switch (_ct_enum)
    {
    case CT_15:
        _thold = 15;
        break;
    case CT_25:
        _thold = 25;
        break;
    case CT_35:
        _thold = 35;
        break;
    case CT_45:
        _thold = 45;
        break;
    case CT_55:
        _thold = 55;
        break;
    case CT_65:
        _thold = 65;
        break;
    case CT_75:
        _thold = 75;
        break;
    case CT_85:
        _thold = 85;
        break;
    default:
        _thold = 85;
        break;
    }

    cannyThreshold = _thold;

    //    std::cout<<"Canny lower Threshold: "<<_thold<<std::endl;

}

void MDADescriptor::setModuleSizeToleranceFromGenome(int _t_enum){ //2807 cambio NE

    int _valueTol =0;

    switch (_t_enum)
    {
    case PCT_10:
        _valueTol = 10;
        break;
    case PCT_20:
        _valueTol = 20;
        break;
    case PCT_30:
        _valueTol = 30;
        break;
    case PCT_40:
        _valueTol = 40;
        break;
    case PCT_50:
        _valueTol = 50;
        break;
    case PCT_60:
       _valueTol = 60;
        break;
    case PCT_70:
        _valueTol = 70;
        break;
    case PCT_80:
        _valueTol = 80;
        break;
    default:
        _valueTol = 50;
        break;
    }

   tolerance = _valueTol;
}


string MDADescriptor::writeOutGenomeSequence(std::vector<int> _genome_vec){

    stringstream ss_genom;
    std::copy( _genome_vec.begin(), _genome_vec.end(), ostream_iterator<int>(ss_genom, ""));
    string string_out = ss_genom.str();
    string_out = string_out.substr(0, string_out.length());

    return string_out;
}
