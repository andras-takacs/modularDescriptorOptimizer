#include "Evaluation/SegmentationEvaluation.h"
#include <QVector>

SegmentationEvaluation::SegmentationEvaluation(){

    mark_keypoints = false;
    matcher_type = BRUTE_FORCE;
    evaluation_type = ROTATE;
    descriptor_type = SIFT_D;
    comparation_method = DESCRIPTOR_INDEX;
    maxGamma = 20;
    minGamma = 0.1;
    destinationTriplet = Point2f(3);

}

SegmentationEvaluation::~SegmentationEvaluation(){

}

void SegmentationEvaluation::setVariables(int _desc_type, int _matcher_type, int _evaluation_type, int _comparation_method){

    evaluation_type = _evaluation_type;
    descriptor_type = _desc_type;
    matcher_type = _matcher_type;
    comparation_method = _comparation_method;

}


void SegmentationEvaluation::setIlluminationVariables(int _desc_type, int _matcher_type, int _max_gamma, int _min_gamma){

    evaluation_type = ILLUMINATION;
    descriptor_type = _desc_type;
    matcher_type = _matcher_type;
    maxGamma =_max_gamma;
    minGamma = _min_gamma;

}

void SegmentationEvaluation::markKeyPoints(){

    mark_keypoints = true;
}


void SegmentationEvaluation::evaluation(vector<Images> &_eval_images)
{
    //    double alphas[13] = {0.33,0.4,0.5,0.67,0.8,0.9,1.0,1.1,1.2,1.5,2.0,2.5,3.0};
    //    double alpha_triplets[18][3] ={{1.000,0.494,0.190}, //3200 K
    //                                   {1.000,0.531,0.232}, //3400 K
    //                                   {1.000,0.567,0.276}, //3600 K
    //                                   {1.000,0.602,0.322}, //3800 K
    //                                   {1.000,0.635,0.370}, //4000 K
    //                                   {1.000,0.666,0.419}, //4200 K
    //                                   {1.000,0.696,0.468}, //4400 K
    //                                   {1.000,0.724,0.519}, //4600 K
    //                                   {1.000,0.752,0.569}, //4800 K
    //                                   {1.000,0.778,0.620}, //5000 K
    //                                   {1.000,0.802,0.671}, //5200 K
    //                                   {1.000,0.826,0.721}, //5400 K
    //                                   {1.000,0.848,0.771}, //5600 K
    //                                   {1.000,0.870,0.820}, //5800 K
    //                                   {1.000,0.890,0.869}, //6000 K
    //                                   {1.000,0.909,0.917}, //6200 K
    //                                   {1.000,0.928,0.965}, //6400 K
    //                                   {0.989,0.935,1.000}};//6600 K
    //    double kelvins[18] = {3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800,
    //                          5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600};


    ////    double affine_triplet[10][3]={{0,0,0},{0.1,0.5,0.15},{0.2,0.1,0.25},{0.3,0.15,0.35},{0.35,0.20,0.45},
    ////                                  {0.45,0.25,0.55},{0.55,0.30,0.65},{0.65,0.35,0.75},{0.70,0.40,0.75},{0.70,0.45,0.75}};

    //    double affine_triplet[10][3]={{0.0,0.9,1.0},{0.0,0.8,1.0},{0.0,0.7,1.0},{0.0,0.6,1.0},{0.0,0.5,1.0}
    //                                  ,{0.1,1.0,1.0},{0.2,1.0,1.0},{0.3,1.0,1.0},{0.4,1.0,1.0},{0.9,0.9,1.0}};

    //    int rot_angles[25] ={0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360};

    //    double sizes[9] = {0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.5};


    //    //    int betas[9] = {-20,-15,-10,-5,0,5,10,15,20};
    //    int betas[9] = {-100,-75,-50,-25,0,25,50,75,100};
    //    int kernel_size[10] = {1,3,5,7,9,11,13,15,17,19};
    int average_i, average_j;
    int enlargement = 21;

    EvaluationValues ev_values;

    clock_t clk1, clk2, clk3;

    int start_iteration=0, end_iteration=0;

    average_i = _eval_images.size();


    if (evaluation_type==ROTATE){

        start_iteration = 0;
        end_iteration = 25;

    }else if (evaluation_type==AFFINE){

        start_iteration = 0;
        end_iteration = 10;

    }else if (evaluation_type==SIZE){

        start_iteration = 0;
        end_iteration = 9;

    }else if (evaluation_type==ILLUMINATION){

        start_iteration = minGamma*10;
        end_iteration = maxGamma*10;

    }else if (evaluation_type==BLUR){

        start_iteration = 0;
        end_iteration = 10;

    }else if(evaluation_type==INTENSITY_CHANGE){
        //! With alpha increment
        start_iteration = 0;
        end_iteration = 13;

    }else if(evaluation_type==INTENSITY_SHIFT){
        //! With beta increment
        start_iteration = 0;
        end_iteration = 9;

    }else if(evaluation_type==INTENSITY_TEMP){
        //! With beta increment
        start_iteration = 0;
        end_iteration = 18;
    }else if(evaluation_type==TILDE_EVAL){

        start_iteration = 1;
        end_iteration = _eval_images.size();
    }

    average_j = end_iteration;

    if(evaluation_type==TILDE_EVAL){

        average_i=1;
    }


    Mat res_y = Mat::zeros(average_i,average_j,CV_64FC1);
    Mat time_1 = Mat::zeros(1,1,CV_32FC1);

    int final_image_number = _eval_images.size();


    if(evaluation_type==TILDE_EVAL){

        final_image_number=1;
    }



    for(int im_it=0;im_it<final_image_number;++im_it){

        Images imgs_1, imgs_2;
        Mat transf_mat, im_for_desc1, im_for_desc2;

        imgs_1 = _eval_images.at(im_it);

        if(evaluation_type==TILDE_EVAL){

            imgs_1 = _eval_images.at(0);
        }



        for(int it=start_iteration; it<end_iteration;++it){


            if (evaluation_type==ROTATE ||evaluation_type==AFFINE || evaluation_type==SIZE){

                if(evaluation_type==AFFINE){
                    transf_mat = ImageProcessing::getAffineTransformMat(imgs_1.col_im.rows,imgs_1.col_im.cols,ev_values.affineTripletAt(it));
                    //resize image box for the transformed image
                    imgs_2 = imgs_1.affineTransformImagesA(transf_mat,ev_values.affineTripletAt(it));
                }else if(evaluation_type==ROTATE){

                    //                    if(descriptor_type=="SPECIAL"){
                    imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(_eval_images.at(im_it),evaluation_type);
                    //                    imgs_2 = ImageProcessing::enlargeImageForRotationAndSize(_eval_images.at(im_it));
                    //                    }+
                    transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5), ev_values.rotationAt(it), 1 );
                    imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);

                }else if(evaluation_type==SIZE){

                    //                    if(descriptor_type=="SPECIAL"){
                    imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(_eval_images.at(im_it),evaluation_type);
                    //                    imgs_2 = ImageProcessing::enlargeImageForRotationAndSize(_eval_images.at(im_it));
                    //                    }

                    transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5), 0, ev_values.sizeAt(it) );
                    imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);
                }

            }else if(evaluation_type==ILLUMINATION){

                imgs_2 = imgs_1.gammaCorrectImages(double(it/10));

            }else if(evaluation_type==BLUR){

                cv::GaussianBlur(imgs_1.col_im,imgs_2.col_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                cv::GaussianBlur(imgs_1.grey_im,imgs_2.grey_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                imgs_1.mask_im = imgs_2.mask_im;

            }else if(evaluation_type==INTENSITY_CHANGE){

                imgs_2.col_im = ImageProcessing::lightIntensityChangeAndShift(imgs_1.col_im, ev_values.alphaAt(it), 0);
                cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

            }else if(evaluation_type==INTENSITY_SHIFT){

                imgs_2.col_im = ImageProcessing::lightIntensityChangeAndShift(imgs_1.col_im,1, ev_values.betaAt(it));
                cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

            }else if(evaluation_type==INTENSITY_TEMP){

                imgs_2.col_im = ImageProcessing::colorTemperatureChange(imgs_1.col_im,ev_values.alphaTripletAt(it), 0);
                cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

            }else if(evaluation_type==TILDE_EVAL){

                imgs_2 = _eval_images.at(it);
            }

            //!-- Step 1: Detect the keypoints using SURF Detector

            //!=========================================HESSIAN KEYPOINT=====================================================
            int minHessian = 400;

            //SurfFeatureDetector detector( minHessian );
            Ptr<SURF> detector = SURF::create( minHessian );

            std::vector<KeyPoint> keypoints_1, keypoints_2;
            std::vector<KeyPoint> keyoints_forT;

            int radius = 10;

            //!=====================================PATCH RADIUS DISTANCE====================================================

            int step = radius*2+1;
            int first = radius+enlargement;
            int last = radius+enlargement;// - 1;

            for (int x=first+imgs_1.rot_offset_x; x< (imgs_1.col_im.cols - last-imgs_1.rot_offset_x); x=x+step){

                for (int y=first+imgs_1.rot_offset_y; y < (imgs_1.col_im.rows - last-imgs_1.rot_offset_y); y=y+step){
                    keypoints_1.push_back( cv::KeyPoint(x,y,8));

                    if (evaluation_type==ROTATE ||evaluation_type==AFFINE ||evaluation_type==SIZE){
                        //! for affine transformation and rotation
                        KeyPoint point = ImageProcessing::pointTransform(cv::KeyPoint(x,y,8),transf_mat);
                        keypoints_2.push_back(point);
                    }else{
                        keypoints_2.push_back( cv::KeyPoint(x,y,8));
                    }
                }
            }


            if(mark_keypoints){
                //! to color the keypoints on the transformed image
                for (uint i=0; i<keypoints_2.size();++i){

                    cv::circle( imgs_2.col_im, cv::Point(keypoints_2[i].pt.x, keypoints_2[i].pt.y), 1, cv::Scalar(255,0, 0) );
                    //! Keypoints on the original image
                    cv::circle( imgs_1.col_im, cv::Point(keypoints_1[i].pt.x, keypoints_1[i].pt.y), 1, cv::Scalar(255,0, 0) );

                }
                //                std::cout<<"Keypoint to draw: "<<keypoints_2[0].pt.x<<", "<<keypoints_2[0].pt.y<<std::endl;
                imshow("Transformed points at size: x"+to_string(ev_values.sizeAt(it)),imgs_2.col_im);

                waitKey();
            }

            //!-- Step 2: Calculate descriptors (feature vectors)


            //            if(descriptor_type=="SIFT"||descriptor_type=="SURF"){
            //                im_for_desc1 = imgs_1.grey_im;
            //                im_for_desc2 = imgs_2.grey_im;
            //            }else{
            im_for_desc1 = imgs_1.col_im;
            im_for_desc2 = imgs_2.col_im;
            //            }

            Descriptor descriptors_1, descriptors_2;

            clk1 = clock();
            descriptors_1.descriptor_feature(keypoints_1,im_for_desc1,descriptor_type, 10);
            clk2 = clock();
            //                    std::cout<<"First image descriptors done"<<std::endl;
            descriptors_2.descriptor_feature(keypoints_2,im_for_desc2,descriptor_type, 10);
            clk3 = clock();
            //        std::cout<<"Second image descriptors done"<<std::endl;

            //            std::cout<<"First img descriptor time: "<<(float)(clk2-clk1)/CLOCKS_PER_SEC<<" sec, Second: "<<(float)(clk3-clk2)/CLOCKS_PER_SEC<< "sec."<<std::endl;

            time_1.push_back( (float)(clk2-clk1)/CLOCKS_PER_SEC);
            time_1.push_back( (float)(clk3-clk2)/CLOCKS_PER_SEC);
            //-- Step 3: Matching descriptor vectors using FLANN matcher


            std::vector< DMatch > matches, good_matches;
            std::vector<std::vector< DMatch > >knn_matches;

            //            //! Normalize descriptor vector before matching


            //            Mat comp_desc1, comp_desc2;

            //            for( int i = 0; i < descriptors_1.descriptors.rows; i++ ){

            //                Mat norm_desc1,norm_desc2;

            //                normalize(descriptors_1.descriptors.row(i),norm_desc1);
            //                normalize(descriptors_2.descriptors.row(i),norm_desc2);

            //                comp_desc1.push_back(norm_desc1);
            //                comp_desc2.push_back(norm_desc2);

            //            }

            if(matcher_type==FLANN){

                FlannBasedMatcher matcher;
                matcher.match( descriptors_1.descriptors, descriptors_2.descriptors, matches );
                //                matcher.match( comp_desc1, comp_desc2, matches );

            }
            else if(matcher_type==BRUTE_FORCE){

                BFMatcher matcher(NORM_L2, true);
                matcher.knnMatch( descriptors_1.descriptors, descriptors_2.descriptors, knn_matches,1);
                //                matcher.knnMatch( comp_desc1, comp_desc2, knn_matches,1);

            }else if(matcher_type==RAW_EUCLID){

                for( int i = 0; i < descriptors_1.descriptors.rows; i++ )
                {
                    Mat first_dot_vec, second_dot_vec;
                    DMatch dmatch;

                    first_dot_vec = descriptors_1.descriptors.row(i);
                    second_dot_vec = descriptors_2.descriptors.row(i);



                    //                    cv::normalize(descriptors_1.descriptors.row(i),first_dot_vec);
                    //                    cv::normalize(descriptors_2.descriptors.row(i),second_dot_vec);

                    //                    std::cout<<"First Vector: "<<first_dot_vec<<std::endl;
                    //                    std::cout<<"Second Vector: "<<second_dot_vec<<std::endl;

                    //                    dot_desc = acos(first_dot_vec.dot(second_dot_vec))*180/PI;

                    //                    dot_desc = first_dot_vec.dot(second_dot_vec);

                    double euclideanDistance = 0;

                    for (int d_i=0;d_i<first_dot_vec.cols;++d_i){

                        double tmp = (double)first_dot_vec.at<float>(0,d_i)-(double)second_dot_vec.at<float>(0,d_i);

                        euclideanDistance +=tmp*tmp;

                    }

                    euclideanDistance = sqrt(euclideanDistance);

                    first_dot_vec.release();second_dot_vec.release();

                    dmatch.distance = (float)euclideanDistance;
                    dmatch.imgIdx = 0;
                    dmatch.queryIdx = i;
                    dmatch.trainIdx = i;

                    matches.push_back(dmatch);
                }
            }

            else if(matcher_type == RAW_EUCLID_KNN){

                for( int i = 0; i < descriptors_1.descriptors.rows; i++ )
                {
                    Mat first_euc_vec;
                    DMatch dmatch;

                    first_euc_vec = descriptors_1.descriptors.row(i);

                    dmatch = euclidWithKnnSearch(first_euc_vec,i,descriptors_2.descriptors);

                    first_euc_vec.release();

                    matches.push_back(dmatch);

//                    std::cout<<i+1<<" descriptor compared. The result distance: "<<dmatch.distance<<", q_idx: "<<dmatch.queryIdx<<" t_idx: "<<dmatch.queryIdx<<std::endl;

                }
            }

            double max_dist = 0; double min_dist = 9999999999;

            //!-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < descriptors_1.descriptors.rows; i++ )
            {
                double dist = 0;
                if(matcher_type==BRUTE_FORCE){
                    dist = knn_matches[i][0].distance;

                    //                    std::cout<<"Image index: "<<knn_matches[i][0].imgIdx<<std::endl;
                }else if (matcher_type==FLANN){
                    dist = matches[i].distance;
                }else if (matcher_type==RAW_EUCLID){
                    dist = matches[i].distance;

                }else if(matcher_type == RAW_EUCLID_KNN){
                    dist = matches[i].distance;
                }

                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;

            }


//            printf("-- Max dist : %f \n", max_dist );
//            printf("-- Min dist : %f \n", min_dist );

            //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
            //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
            //-- small)
            //-- PS.- radiusMatch can also be used here.


            int desc_size = descriptors_1.descriptors.rows;


            //            std::cout<<"Vec Degree: "<<vec_degree.rows<<" Descriptor Size: "<<desc_size<<std::endl;
            for( int i = 0; i < desc_size; i++ )
            {
                //                if( knn_matches[i][0].distance <=  0.02 /*max(min_dist*5, 0.02)*/) {
                //                    good_matches.push_back(knn_matches[i][0]);
                //                }

                if(comparation_method==DESCRIPTOR_INDEX){
                    if((matcher_type==BRUTE_FORCE) && (knn_matches[i][0].queryIdx == knn_matches[i][0].trainIdx)){
                        good_matches.push_back(knn_matches[i][0]);

                    }else if((matcher_type==FLANN)&&(matches[i].queryIdx == matches[i].trainIdx)){

                        good_matches.push_back(matches[i]);
                    }else if(matcher_type == RAW_EUCLID){
                        double tmp_dist = 0.0;
                        double range =  max_dist-min_dist;

                        if(range ==0){
                            tmp_dist = 0;
                        }else{
                            tmp_dist = (float)matches[i].distance/range;
                        }
                        if(tmp_dist<0.05){

                            good_matches.push_back(matches[i]);
                        }


                    }else if(matcher_type == RAW_EUCLID_KNN&&(matches[i].queryIdx == matches[i].trainIdx)){

                        good_matches.push_back(matches[i]);
                    }
                }
            }
            //        printf("Good Matches: %f, Decriptor number: %f, Percent of good matches: %f%% \n", double(good_matches.size()),double(descriptors_1.descriptors.rows),(double(good_matches.size())/double(descriptors_1.descriptors.rows)) );
            //            std::cout<<"At round no. "<<it<<" the Good Matches: "<<double(good_matches.size())/double(descriptors_1.descriptors.rows)<<std::endl;
            //            std::cout<<"Image no.: "<<im_it<<" at round: "<<it<<" the min distance is: "<< min_dist<<" and max dist: "<<max_dist<<std::endl;

            //!        -- Draw only "good" matches======================================
            //            if(evaluation_type!=ROTATION){
            if(it==(end_iteration-1)){
                drawMatches( imgs_1.col_im, keypoints_1, imgs_2.col_im, keypoints_2,
                             good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                             vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                altered_im = imgs_2.col_im;
            }
            //            }
            //!=========================================================================

            //                        imshow("Good Matches", img_matches);
            //                        waitKey();

            if(evaluation_type==INTENSITY_SHIFT){

                x_vector.push_back(double(ev_values.betaAt(it)));
            }
            else if(evaluation_type==BLUR){

                x_vector.push_back(double(ev_values.kernelSizeAt(it)));
            }
            else if(evaluation_type==INTENSITY_TEMP){

                x_vector.push_back(ev_values.kelvinAt(it));
            }
            else if(evaluation_type==ROTATE){

                x_vector.push_back(double(ev_values.rotationAt(it)));
            }
            else
            {
                x_vector.push_back(double(it));
            }

            y_vector.push_back(double(good_matches.size())/double(descriptors_1.descriptors.rows)*100);
            res_y.at<double>(im_it,it) = double(good_matches.size())/double(descriptors_1.descriptors.rows)*100;



            //!-- Show detected matches
        }

        std::cout<<"Image no. "<<im_it<<" Match vector: "<<res_y<<std::endl;

        x_average_vector.push_back(x_vector);
        y_average_vector.push_back(y_vector);




        //        imshow( "Good Matches", img_matches );

    }

    //    Mat x_results = Mat::zeros(average_i,average_j,CV_64FC1);
    QVector<double> x(average_j),y(average_j);
    Mat av_y, av_time;
    cv::reduce(res_y,av_y,0,CV_REDUCE_AVG);
    cv::reduce(time_1,av_time,0,CV_REDUCE_AVG);
    descriptor_count_time = double(av_time.at<float>(0,0));

    std::cout<<"Final average vector: "<<av_y<<std::endl;
    std::cout<<"Final average descriptor time: "<<descriptor_count_time<<std::endl;

    for (int av_j=0; av_j<average_j; ++av_j){

        y[av_j] = av_y.at<double>(0,av_j);

    }

    plot_vector.push_back(x_average_vector.at(0));
    plot_vector.push_back(y);
}


DMatch SegmentationEvaluation::euclidWithKnnSearch(Mat &baseVector, int baseVectorAt, Mat &compareMat){

    DMatch result;


    Mat first_euc_vec, second_euc_mat,subt_mat,sqr_mat,sum_mat,sqrt_mat;

    first_euc_vec = baseVector;
    second_euc_mat = compareMat;

    double euclideanDistance = 0;
    double shortestDistnce = 99999999999999;
    int bestIndex = 0;

    for(int q_i=0;q_i<second_euc_mat.rows-1;++q_i){
        first_euc_vec.push_back(baseVector);
    }


    cv::subtract(first_euc_vec,second_euc_mat,subt_mat);

    cv::pow(subt_mat,2.0,sqr_mat);

    cv::reduce(sqr_mat,sum_mat,1,CV_REDUCE_SUM);

    cv::sqrt(sum_mat,sqrt_mat);




    for(int q_i=0;q_i<second_euc_mat.rows;++q_i){

        euclideanDistance = sqrt_mat.at<float>(q_i,0);

        if(shortestDistnce>euclideanDistance){

            bestIndex = q_i;
            shortestDistnce = euclideanDistance;
        }
    }



    result.distance = (float)shortestDistnce;
    result.imgIdx = 0;
    result.queryIdx = baseVectorAt;
    result.trainIdx = bestIndex;

    first_euc_vec.release();second_euc_mat.release();subt_mat.release();sqr_mat.release();sum_mat.release();sqrt_mat.release();

    return result;

}

