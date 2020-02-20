#include "Segmentation/BoundingBox.h"

using namespace cv;
using namespace std;

BoundingBox::BoundingBox()
{

}

BoundingBox::~BoundingBox()
{

}

void BoundingBox::get_bounding_box(cv::Mat &in_ext, cv::Mat &col_in){
    RNG rng(12345);
    cv::Mat ext_gray;
    int dilation_size=3;
    cv::cvtColor(in_ext,ext_gray,CV_BGR2GRAY);

    //Get structuring element for dilation and erode
    cv::Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    // Apply the dilation operation
    cv::erode ( ext_gray,ext_gray, element );
    cv::dilate ( ext_gray,ext_gray, element );

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;


    //Get contours and fitting rectangle
    findContours( ext_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    for( uint i = 0; i < contours.size(); i++ )
    { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }


    // Draw contours and bounding box
    for( uint i = 0; i< contours.size(); i++ )
    {
        if(boundRect[i].area()>5000 && boundRect[i].y<in_ext.rows*0.75 ){
            std::cout<<"AREA at "<<i<<": "<<boundRect[i].area()<<std::endl;
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            cv::drawContours( in_ext, contours, i, color, 2, 8, hierarchy, 0, Point() );
            cv::rectangle( col_in, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        }
    }
}

