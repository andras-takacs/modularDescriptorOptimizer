#include "Images/Images.h"

Images::Images()
{
    col_im = cv::Mat(1,1,CV_8UC1,cv::Scalar::all(0));
    grey_im = cv::Mat(1,1,CV_8UC1,cv::Scalar::all(0));
    mask_im = cv::Mat(1,1,CV_8UC1,cv::Scalar::all(0));

    rot_offset_x = 0;
    rot_offset_y = 0;

}

Images::~Images()
{
    col_im.release();
    grey_im.release();
    mask_im.release();
}

Images::Images(const cv::Size _mat_size)
{
    col_im = cv::Mat(_mat_size,CV_8UC3,cv::Scalar::all(0));
    grey_im = cv::Mat(_mat_size,CV_8UC1,cv::Scalar::all(0));
    mask_im = cv::Mat(_mat_size,CV_8UC1,cv::Scalar::all(0));

    rot_offset_x = 0;
    rot_offset_y = 0;

}


void Images::loadImages(vector<string> _im_data)
{

    cv::Mat grey, src, mask_src;
    string mask_="mask_";
    string fileName = _im_data.at(0)+ _im_data.at(2);
    string mfileName = _im_data.at(1)+"/"+_im_data.at(2);
    char cfilename[200];
    char mcfilename[200];
    strcpy(cfilename,fileName.c_str());
    strcpy(mcfilename,mfileName.c_str());

    const char* filename = cfilename;
    const char* maskfilename = mcfilename;

    src = imread(filename, 1);
    mask_src = imread(maskfilename,cv::IMREAD_GRAYSCALE);

    if ( src.empty() || mask_src.empty())
    {
        if(src.empty()){
            std::cerr << "Image "<<_im_data.at(2)<<" hasn't been loaded!' "<< std::endl;
        }else if(mask_src.empty()){
            std::cerr << "Mask of "<<_im_data.at(2)<<" hasn't been loaded!' "<< std::endl;
        }
        //return 1;
    }

    if ( src.channels() > 1 ){
        cv::cvtColor( src, grey, CV_BGR2GRAY );
    }
    else{
        src.copyTo(grey);
    }

    std::cout<<filename<<std::endl;
    std::cout<<maskfilename<<std::endl;

    col_im = src;
    mask_im = mask_src;
    grey_im = grey;

    src.release();
    mask_src.release();
    grey.release();

    std::cout<<"Finished reading the image triplet"<<std::endl;
}

Images Images::affineTransformImagesA(Mat& _aff_transform_mat, const std::vector<double> &_affine_triplet){


    //    std::cout<<"Transformation map: "<<_aff_transform_mat<<std::endl;

    Images warp_dest_im;

    Mat _col_im, _grey_im, _mask_im;

    _col_im = col_im;
    _grey_im = grey_im;
    _mask_im = mask_im;

    //    std::cout<<"New height: "<<_affine_triplet[1]<<std::endl;
    /// Set the dst image the same type and size as src
    warp_dest_im.col_im = cv::Mat::zeros( (_col_im.rows+_col_im.rows*(1-_affine_triplet[1])), _col_im.cols, _col_im.type() );
    warp_dest_im.grey_im = cv::Mat::zeros( _grey_im.rows, _grey_im.cols, _grey_im.type() );
    warp_dest_im.mask_im = cv::Mat::zeros( _mask_im.rows, _mask_im.cols, _mask_im.type() );


    /// Apply the Affine Transform just found to the src image
    warpAffine( _col_im,  warp_dest_im.col_im, _aff_transform_mat,  warp_dest_im.col_im.size() );
    warpAffine( _grey_im,  warp_dest_im.grey_im, _aff_transform_mat,  warp_dest_im.grey_im.size() );
    warpAffine( _mask_im,  warp_dest_im.mask_im, _aff_transform_mat,  warp_dest_im.mask_im.size() );

    return warp_dest_im;

}

Images Images::affineTransformImagesRS(Mat& _aff_transform_mat){


    //    std::cout<<"Transformation map: "<<_aff_transform_mat<<std::endl;

    Images warp_dest_im(col_im.size());

    Mat _col_im, _grey_im, _mask_im;
    _col_im = cv::Mat(col_im.size(),CV_8UC3,cv::Scalar::all(0));
    _grey_im = cv::Mat(col_im.size(),CV_8UC1,cv::Scalar::all(0));
    _mask_im = cv::Mat(col_im.size(),CV_8UC1,cv::Scalar::all(0));

    _col_im = col_im;
    _grey_im = grey_im;
    _mask_im = mask_im;

    /// Set the dst image the same type and size as src
    //    warp_dest_im.col_im = cv::Mat::zeros( _col_im.rows, _col_im.cols, _col_im.type() );
    //    warp_dest_im.grey_im = cv::Mat::zeros( _grey_im.rows, _grey_im.cols, _grey_im.type() );
    //    warp_dest_im.mask_im = cv::Mat::zeros( _mask_im.rows, _mask_im.cols, _mask_im.type() );



    /// Apply the Affine Transform just found to the src image
    warpAffine( _col_im,  warp_dest_im.col_im, _aff_transform_mat,  warp_dest_im.col_im.size() );
    warpAffine( _grey_im,  warp_dest_im.grey_im, _aff_transform_mat,  warp_dest_im.grey_im.size() );
    warpAffine( _mask_im,  warp_dest_im.mask_im, _aff_transform_mat,  warp_dest_im.mask_im.size() );

    return warp_dest_im;

}

Images Images::gammaCorrectImages(double gamma){

    Images _returnImages;

    _returnImages.col_im = ImageProcessing::gammaCorrection(col_im, gamma);
    _returnImages.grey_im = ImageProcessing::gammaCorrection(grey_im, gamma);
    _returnImages.mask_im = ImageProcessing::gammaCorrection(mask_im, gamma);


    return _returnImages;

}


void Images::getImagesFromDatabase(Database &db, int position){


    Mat src, grey, label_src;
    bool itHasLabel = true;

    string image_extension = ".jpg";
    string label_extension = ".png";

    if(db.database_id==BRIGHTON_DB){
        image_extension = ".pnm";
        label_extension = ".pnm";
    }else if (db.database_id==LABELME_FACADE_DB_JENA || db.database_id==LABELME_FACADE_DB_ALL){
        image_extension = ".jpg";
        label_extension = ".png";

    }else if (db.database_id==TILDE_DB||db.database_id==OXFORD_DB){
        image_extension = ".png";
        label_extension = ".png";

    }

    string fileName = db.root_folder+db.images_folder_name+db.image_list[position]+image_extension;
    string lfileName = db.root_folder+db.labels_folder_name+db.image_list[position]+label_extension;
    char cfilename[200];
    char lcfilename[200];
    strcpy(cfilename,fileName.c_str());
    strcpy(lcfilename,lfileName.c_str());

    //std::cout<<"number of images: "<<image_list.size()<<std::endl;
    //for(int i=0;i<image_list.size();++i){
    //std::cout<<"Image name: "<<image_list[i]<<"\n";
    //}

    const char* imagefilename = cfilename;
    const char* labelfilename = lcfilename;

    src = imread(imagefilename, 1);

    if (db.database_id==OXFORD_DB){

        itHasLabel=false;
    }


    if (itHasLabel){
        if(db.database_id==BRIGHTON_DB || db.database_id==TILDE_DB){
            label_src = imread(labelfilename, cv::IMREAD_GRAYSCALE);
        }else if (db.database_id==LABELME_FACADE_DB_JENA || db.database_id==LABELME_FACADE_DB_ALL){
            label_src = imread(labelfilename, 1);
        }
    }

//    std::cout<<"Got until: "<<labelfilename<<" channels: "<<src.channels()<<std::endl;

    if ( src.empty() || label_src.empty())
    {
        if(src.empty()){
            std::cerr << "Image "<<imagefilename<<" hasn't been loaded!' "<< std::endl;
        }if(label_src.empty() && itHasLabel){
            std::cerr << "Label of "<<labelfilename<<" hasn't been loaded!' "<< std::endl;
        }
    }


    cv::cvtColor(src,src,CV_BGR2RGB); //converting opencv BGR image to RGB

    if ( src.channels() > 1 ){
        cv::cvtColor( src, grey, CV_RGB2GRAY );
    }
    else{
        src.copyTo(grey);
    }

    col_im = src;
    mask_im = label_src;
    grey_im = grey;

    src.release();
    label_src.release();
    grey.release();

//    std::cout<<"At "<<position<<" finished reading the image triplet of "<<db.image_list[position]<<std::endl;

}



