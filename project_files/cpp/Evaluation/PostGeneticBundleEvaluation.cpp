#include "Evaluation/PostGeneticBundleEvaluation.h"


PostGeneticBundleEvaluation::PostGeneticBundleEvaluation()
{

}
PostGeneticBundleEvaluation::PostGeneticBundleEvaluation(const int& _database, int _eval_im_number, int setComputer)
{
    usedDatabase = _database;
    mark_keypoints = false;
    number_of_evaluation_images = _eval_im_number;
    matcher_type = BRUTE_FORCE; //It can be FLANN or BRUTE_FORCE
    comparation_method=DESCRIPTOR_INDEX; // COMPARATION METHOD: "DOT_PRODUCT", "DESCRIPTOR_INDEX"
    usedComputer = setComputer;
    weight = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
}

PostGeneticBundleEvaluation::~PostGeneticBundleEvaluation()
{

}

void PostGeneticBundleEvaluation::evaluate()
{


    calculateModularDescriptor();

    trainingEvaluation(8);


}

void PostGeneticBundleEvaluation::calculateModularDescriptor(){



    std::shared_ptr<MDADescriptor> mDescriptor (new MDADescriptor());
    evaluatedDescriptor = mDescriptor;

    evaluatedDescriptor->setGenome(modularGenome);

    //    MDADescriptor mDescriptor(bin_genome_vec);

    DescriptorAssembler modularDescriptor(evaluatedDescriptor,FINAL);

    modularDescriptor.buildUp();

}

void PostGeneticBundleEvaluation::trainingEvaluation(int desc_round){

    cv::Mat total_desc,total_label;
    int loop_start,loop_finish;
    int patch_radius = evaluatedDescriptor->getPatchRadius();

    //Set training image database
    Database im_db;
    im_db.setDataBase(usedDatabase, usedComputer, TRAIN);

    loop_start =0;
    loop_finish = im_db.size;
    //    loop_finish = 2;

    std::cout<<"Loop start: "<<loop_start<<" Loop finish: "<<loop_finish<<std::endl;

    //Load images from database
    Images ref_im;
    ref_im.getImagesFromDatabase(im_db,0);

    //Instantiate keypoint class
    ImageKeypoints image_key_points(PATCH_DIST_KP,ref_im.col_im,0,patch_radius);
    image_key_points.calculateCoordinates(ref_im);

    std::cout<<"Loaded all images!"<<std::endl;

    //Intantiate modular descriptor class
    DescriptorCalculator modularDescriptor(evaluatedDescriptor);
    ImageLabeling labelExtractor(PROJECT_4_CLASS);



    //CALCULATE MODULAR DESCRIPTOR AND GET LABEL CLASS FOR TRAINING
    for(int t=loop_start;t<loop_finish;++t)
    {
        Mat desc_result, label_result;

        Images im;
        im.getImagesFromDatabase(im_db,t);

        if(image_key_points.theImageIsDifferent(im.col_im)){
            image_key_points.calculateCoordinates(im);
        }
        vector<KeyPoint> train_kp = image_key_points.getKeyPoints();


        if (desc_round<8){

            Descriptor descriptors_1;

            descriptors_1.descriptor_feature(train_kp,im.col_im,desc_round);
            desc_result = descriptors_1.descriptors;


        }else if (desc_round==8){


            desc_result = modularDescriptor.calculate(im,train_kp);

        }


        label_result = labelExtractor.labelingImage(im,train_kp);
        total_desc.push_back(desc_result);
        total_label.push_back(label_result);

    }

    std::cout<<"Extracted descriptor matrix size: "<<total_desc.cols<<", "<<total_label.rows<<std::endl;

    //INSTANTIATE RandomTree for train and test
    //    RandomForest* rtree = new RandomForest();
    std::unique_ptr<RandomForest> rtree(new RandomForest());

    std::cout<<"Start tree tarining!"<<std::endl;

    int rf_depth = evaluatedDescriptor->getTreeDepth();
    int rf_size = evaluatedDescriptor->getTreeNumbers();
    int rf_split = evaluatedDescriptor->getSplitNumber();

    //TRAINING the tree
    rtree->train(total_desc,total_label,desc_round,rf_depth,rf_size,rf_split);
    std::cout<<"Finished tree tarining!"<<std::endl;



    im_db.setDataBase(usedDatabase, usedComputer, TEST);

    //TESTING the tree
    int t_loop_start = 0;
    int t_loop_finish = im_db.size;

    cv::Mat all_res_percent, all_measured_times;


    //MEASURING DESCRIPTOR EXTRACTION + BOUNDING BOX FITTING
    std::clock_t descriptor_ext_start;
    double descriptor_ext_duration;

    //TESTING AND BOUNDINGBOX FITTING--->in a parallel thread with pragma
    //#pragma omp parallel for

    cv::Mat classResults;
    for(int tst=t_loop_start;tst<t_loop_finish;++tst)
    {
        cv::Mat measured_times(1,2,CV_64FC1);
        Mat test_descriptors, test_labels;

        Images test_im;
        test_im.getImagesFromDatabase(im_db,tst);

//        if(image_key_points.theImageIsDifferent(test_im.col_im)){
//            image_key_points.calculateCoordinates(test_im);
//        }

        image_key_points.kpType(2);
        image_key_points.calculateCoordinates(test_im);

        descriptor_ext_start = std::clock();

        std::cout<<"Got testing images!"<<std::endl;

        //Calculate the test descriptor and labels
        vector<KeyPoint> test_kp = image_key_points.getKeyPoints();


        if (desc_round<8){

            Descriptor descriptors_1;

            descriptors_1.descriptor_feature(test_kp,test_im.col_im,desc_round);
            test_descriptors = descriptors_1.descriptors;


        }else if (desc_round==8){


            test_descriptors = modularDescriptor.calculate(test_im,test_kp);

        }


        test_labels = labelExtractor.labelingImage(test_im,test_kp);

        std::cout<<"Got testing labels and descriptors!"<<std::endl;

        descriptor_ext_duration = ( std::clock() - descriptor_ext_start ) / (double) CLOCKS_PER_SEC;
        measured_times.at<double>(0,0)=descriptor_ext_duration;

        //Initialize image matrices
        cv::Mat results(test_descriptors.rows,1,CV_32FC1);
        cv::Mat entropy(test_descriptors.rows,1,CV_32FC1);
        cv::Mat res_percent(1,NUMBER_OF_CLASSES+1,CV_32FC1);

        std::cout<<"The results of test at Image "<<tst<<":"<<std::endl;
        rtree->test( test_descriptors,test_labels, results, entropy, res_percent, desc_round);
        colorAndSaveClassResults(test_kp,test_im.col_im,results,entropy,desc_round,tst);

        all_res_percent.push_back(res_percent);

        std::cout<<"No "<<tst+1<<" Test finished."<<std::endl;
        fflush(stdout);
        std::cout<<"No "<<tst+1<<" Round completed"<<std::endl;

        all_measured_times.push_back(measured_times);

        measured_times.release();
        test_descriptors.release();
        test_labels.release();
        results.release();
        entropy.release();
        res_percent.release();
    }

    Mat av_percent, av_times, time_score, percent_score;

    Mat res_t = all_res_percent.t();
    vector<double> stdDevs;
    vector<double> means;

    for(int si=0;si<res_t.rows;++si){
      Scalar mean, stddev;
      meanStdDev ( res_t.row(si), mean, stddev );
      means.push_back(mean[0]);
      stdDevs.push_back(stddev[0]);


    }
    std::cout<<"All Means: "<<means[0]<<", "<<means[1]<<", "<<means[2]<<", "<<means[3]<<std::endl;
    std::cout<<"All Standard deviation: "<<stdDevs[0]<<", "<<stdDevs[1]<<", "<<stdDevs[2]<<", "<<stdDevs[3]<<std::endl;

    cv::reduce(all_res_percent,av_percent,0,CV_REDUCE_AVG,-1);
    cv::reduce(all_measured_times,av_times,0,CV_REDUCE_AVG,-1);

    std::cout<<"The tree test average result: "<<av_percent<<std::endl;

    evaluatedDescriptor->segmentationResults = av_percent;
    evaluatedDescriptor->calculationTimesResult = av_times;

    av_percent = av_percent*0.01;

    av_percent.convertTo(percent_score,CV_64F);
    av_times.convertTo(time_score,CV_64F);


    //Adds the weighted time score and the weighted classification score to the results vector
    double _time_score = time_score.at<double>(0,0)*weight[7];
    score_results.push_back(_time_score);
    evaluatedDescriptor->timeScores.push_back(_time_score);
    std::cout<<"Time Score: "<<time_score.at<double>(0,0)*weight[7]<<std::endl;
    for(int si=0;si<percent_score.cols;++si){
        double _score = 1.0-(percent_score.at<double>(0,si)*weight[si+8]);
        score_results.push_back(_score);
        evaluatedDescriptor->segmentationScores.push_back(_score);
        std::cout<<"Score "<<si<<": "<<_score<<std::endl;
    }
    av_percent.release();
    av_times.release();
    time_score.release();
    percent_score.release();

    std::cout<<"Finished training and testing the "<<getDescriptorName(desc_round)<<" descriptor with Random Trees"<<std::endl;

}



void PostGeneticBundleEvaluation::transformEvaluation2(int desc_round){

    std::vector<Images> evaluation_imgs,ox1_im,ox2_im,ox3_im,til_im;
    vector<Mat> hom_mat1,hom_mat2,hom_mat3;

    //    std::cout<<"Image loading starts"<<std::endl;
    //! Get test images
    Database eval_im_db,ox_db,eval_im2_db;
    eval_im_db.setDataBase(usedDatabase, usedComputer,TRAIN);

    for(int tim_it=0;tim_it<number_of_evaluation_images;++tim_it){
        Images one_im;
        one_im.getImagesFromDatabase(eval_im_db,tim_it);
        evaluation_imgs.push_back(one_im);
    }

    for (int alt_it =0;alt_it<3;++alt_it){

        int subfolder = 0;
        std::vector<Images> pre_evaluation_imgs;
        std::vector<Mat> pre_homographyMats;

        if(alt_it == 0){

            subfolder = LEUVEN;

        }else if(alt_it == 1){

            subfolder = NOTREDAME;

        }else if(alt_it == 2){

            subfolder = UBC;
        }

        ox_db.setDataBase(OXFORD_DB,usedComputer,TEST,subfolder);

        for(int tim_it=0;tim_it<ox_db.size;++tim_it){
            Images one_im;
            one_im.getImagesFromDatabase(ox_db,tim_it);
            pre_evaluation_imgs.push_back(one_im);
        }

        string hom_root = ox_db.root_folder + ox_db.homography_folder_name;
        pre_homographyMats= loadHomography(hom_root, ox_db.size, ox_db.homography_list);

        if(alt_it == 0){
            hom_mat1 =pre_homographyMats;
            ox1_im = pre_evaluation_imgs;

        }else if(alt_it == 1){

            hom_mat2 =pre_homographyMats;
            ox2_im = pre_evaluation_imgs;

        }else if(alt_it == 2){

            hom_mat3 =pre_homographyMats;
            ox3_im = pre_evaluation_imgs;
        }

    }

    eval_im2_db.setDataBase(TILDE_DB,usedComputer,TEST,MEXICO);

    for(int tim_it=0;tim_it<eval_im2_db.size;++tim_it){
        Images one_im;
        one_im.getImagesFromDatabase(eval_im2_db,tim_it);
        til_im.push_back(one_im);
    }



    //    std::cout<<"Images are loaded"<<std::endl;


    EvaluationValues ev_values;

    //Evaluation_type(enumeration value):
    // 0. Rotation(0)
    // 1. Affine Transformation(1)
    // 2. Resize(2)
    // 3. Blur(3)
    // 4. Light Change (4) - Leuven Images from Oxford library
    // 5. Light Condition change (5) - Notre Dame Images from Oxford lib.
    // 6. JPEG compression change (6) - University of British Columbia pictures from Oxford lib.
    // 7. Light Condition change2 (7) - Mexico Images from Tilde lib.

    std::vector<string> eval_name = {"Rotation", "Affine Transformation", "Resize", "Blur", "Light Change", "Light condition change", "JPEG compression", "Light condition change2"};
    std::vector<int> current_eval_max = {ev_values.rotations_vec_length(),
                                         ev_values.affine_vec_length(),
                                         ev_values.sizes_vec_length(),
                                         ev_values.kernels_vec_length(),
                                         5, // Number of images in the Leuven library -1
                                         5, // Number of images in the Notre Dame library -1
                                         5, // Number of images in the UBC ligbrary -1
                                         9  // Number of images in the Mexico ligbrary -1
                                        };


    //run all the evaluation cases---> running parallel with pragma
    //#pragma omp parallel for
    for(int evaluation_type=0;evaluation_type<8;++evaluation_type){

        std::cout<<eval_name[evaluation_type]<<" evaluation starts!"<<std::endl;

        vector<Mat> homographyMats;

        if(evaluation_type==LIGHT_COND_TR2 || evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){

            evaluation_imgs.clear();

            if(evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){


                if(evaluation_type==LIGHT_CH_TR){

                    evaluation_imgs = ox1_im;
                    homographyMats = hom_mat1;

                }else if(evaluation_type==LIGHT_COND_TR){

                    evaluation_imgs = ox2_im;
                    homographyMats = hom_mat2;

                }else if(evaluation_type==JPEG_COMPRESSION){

                    evaluation_imgs = ox3_im;
                    homographyMats = hom_mat3;
                }


            }else if(evaluation_type==LIGHT_COND_TR2){

                evaluation_imgs = til_im;

            }
        }

        int number_of_images, number_of_evaluation_case;

        clock_t clk1, clk2, clk3;

        int start_iteration =0;
        int end_iteration = current_eval_max[evaluation_type];


        //setup values for the result matrix
        number_of_images = evaluation_imgs.size();
        number_of_evaluation_case = end_iteration;


        if(evaluation_type==LIGHT_COND_TR2 || evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){

            number_of_images=1;

        }

        Mat res_y = Mat::zeros(number_of_images,end_iteration,CV_64FC1);
        Mat time_1 = Mat::zeros(1,1,CV_32FC1);



        for(int im_it=0;im_it<number_of_images;++im_it){

            //            std::cout<<"Image "<<im_it<<" is under process!"<<std::endl;

            Images imgs_1, imgs_2;
            Mat transf_mat, im_for_desc1, im_for_desc2;

            imgs_1 = evaluation_imgs.at(im_it);

            if(evaluation_type==LIGHT_COND_TR2 || evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){

                imgs_1 = evaluation_imgs.at(0);
            }

            for(int it=start_iteration; it<end_iteration;++it){

                //If evaluation type is Rotation, Affine Transformation or Resize creates transformation matrix
                if (evaluation_type==ROTATION_TR ||evaluation_type==AFFINE_TR ||evaluation_type==SIZE_TR){

                    if(evaluation_type==ROTATION_TR){

                        imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(evaluation_imgs.at(im_it),ROTATE);

                        transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5),ev_values.rotationAt(it), 1 );

                        imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);

                    }else if(evaluation_type==AFFINE_TR){

                        transf_mat = ImageProcessing::getAffineTransformMat(imgs_1.col_im.rows,imgs_1.col_im.cols,ev_values.affineTripletAt(it));
                        //resize image box for the transformed image
                        imgs_2 = imgs_1.affineTransformImagesA(transf_mat,ev_values.affineTripletAt(it));

                    }else if(evaluation_type==SIZE_TR){

                        imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(evaluation_imgs.at(im_it),SIZE);

                        transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5), 0, ev_values.sizeAt(it));

                        imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);
                    }


                }else if(evaluation_type==BLUR_TR){


                    cv::GaussianBlur(imgs_1.col_im,imgs_2.col_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                    cv::GaussianBlur(imgs_1.grey_im,imgs_2.grey_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                    imgs_1.mask_im = imgs_2.mask_im;

                }else if(evaluation_type==LIGHT_COND_TR2 || evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){

                    imgs_2 = evaluation_imgs.at(it+1);
                }


                //!-- Step 1: Detect the keypoints
                //!========================================= KEYPOINT=====================================================

                int patch_radius = evaluatedDescriptor->getPatchRadius();
                int margin=0;

                if(evaluation_type==ROTATION_TR ||
                        evaluation_type==AFFINE_TR ||
                        evaluation_type==SIZE_TR ||
                        evaluation_type==LIGHT_CH_TR ||
                        evaluation_type==LIGHT_COND_TR ||
                        evaluation_type==JPEG_COMPRESSION){

                    margin = patch_radius*2+1;
                }
                //

                //                imshow("Blured Image",imgs_2.col_im);
                //                imshow("Normal Image",imgs_1.col_im);

                //                waitKey();

                //Instantiate keypoint class

                ImageKeypoints evaluation_keypoints(PATCH_DIST_KP,imgs_1.col_im,margin,patch_radius);

                evaluation_keypoints.calculateCoordinates(imgs_1);

                if(evaluation_type==ROTATION_TR ||evaluation_type==AFFINE_TR ||evaluation_type==SIZE_TR){
                    //! for affine transformation and rotation
                    evaluation_keypoints.transformAllKeypoints(transf_mat);

                }else if(evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){

                    evaluation_keypoints.homographyTransformAllKeypoints(homographyMats[it]);
                }

                if(mark_keypoints){
                    //! to color the keypoints on the transformed image
                    for (uint i=0; i<evaluation_keypoints.keypoints.size();++i){

                        KeyPoint kp_1,kp_2;

                        kp_1 = evaluation_keypoints.keypoints[i];
                        if(evaluation_type==ROTATION_TR ||evaluation_type==AFFINE_TR ||evaluation_type==SIZE_TR ||
                                evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){
                            kp_2 = evaluation_keypoints.transformedKeypoints[i];
                        }

                        cv::circle( imgs_2.col_im, cv::Point(kp_2.pt.x, kp_2.pt.y), 1, cv::Scalar(255,0, 0) );
                        //! Keypoints on the original image
                        cv::circle( imgs_1.col_im, cv::Point(kp_1.pt.x, kp_1.pt.y), 1, cv::Scalar(255,0, 0) );

                    }
                    //                std::cout<<"Keypoint to draw: "<<keypoints_2[0].pt.x<<", "<<keypoints_2[0].pt.y<<std::endl;
                    imshow("Transformed points at size: x"+to_string(ev_values.sizeAt(it)),imgs_2.col_im);

                    waitKey();
                }


                //!-- Step 1: Detect the keypoints
                //!========================================= DESCRIPTOR =====================================================

                im_for_desc1 = imgs_1.col_im;
                im_for_desc2 = imgs_2.col_im;

                //                Descriptor descriptors_1, descriptors_2;

                Mat desc_result_ground_truth, desc_result_transformed;



                //Intantiate modular descriptor class
                DescriptorCalculator modularDescriptor(evaluatedDescriptor);

                //                std::cout<<"Got before descriptor calculation"<<std::endl;
                if (desc_round<8){

                    Descriptor descriptors_1, descriptors_2;

                    clk1 = clock();
                    descriptors_1.descriptor_feature(evaluation_keypoints.keypoints,im_for_desc1,desc_round);
                    desc_result_ground_truth = descriptors_1.descriptors;
                    clk2 = clock();

                    descriptors_2.descriptor_feature(evaluation_keypoints.transformedKeypoints,im_for_desc2,desc_round);
                    desc_result_transformed = descriptors_2.descriptors;
                    clk3 = clock();

                }else if (desc_round==8){

                    clk1 = clock();
                    desc_result_ground_truth = modularDescriptor.calculate(imgs_1,evaluation_keypoints.keypoints);
                    clk2 = clock();
                    desc_result_transformed = modularDescriptor.calculate(imgs_2,evaluation_keypoints.transformedKeypoints);
                    clk3 = clock();
                }


                time_1.push_back( (float)(clk2-clk1)/CLOCKS_PER_SEC);
                time_1.push_back( (float)(clk3-clk2)/CLOCKS_PER_SEC);
                //-- Step 3: Matching descriptor vectors using FLANN matcher


                std::vector< DMatch > matches;
                std::vector<std::vector< DMatch > >knn_matches;


                if(matcher_type==FLANN){

                    FlannBasedMatcher matcher;
                    matcher.match( desc_result_ground_truth, desc_result_transformed, matches );
                    //                matcher.match( comp_desc1, comp_desc2, matches );

                }
                else if(matcher_type==BRUTE_FORCE){

                    BFMatcher matcher(NORM_L2, true);
                    matcher.knnMatch( desc_result_ground_truth, desc_result_transformed, knn_matches,1);
                    //                matcher.knnMatch( comp_desc1, comp_desc2, knn_matches,1);

                }

                double max_dist = 0; double min_dist = 100;

                //!-- Quick calculation of max and min distances between keypoints
                for( int i = 0; i < desc_result_ground_truth.rows; i++ )
                {
                    double dist=0;
                    if(matcher_type==BRUTE_FORCE){
                        dist = knn_matches[i][0].distance;
                    }else if (matcher_type==FLANN){
                        dist = matches[i].distance;
                    }

                    if( dist < min_dist ) min_dist = dist;
                    if( dist > max_dist ) max_dist = dist;

                }




                std::vector< DMatch > good_matches;
                int desc_size = desc_result_ground_truth.rows;

                for( int i = 0; i < desc_result_ground_truth.rows; i++ )
                {


                    if(comparation_method==DESCRIPTOR_INDEX){
                        if((matcher_type==BRUTE_FORCE) && (knn_matches[i][0].queryIdx == knn_matches[i][0].trainIdx)){
                            good_matches.push_back(knn_matches[i][0]);

                        }else if((matcher_type==FLANN)&&(matches[i].queryIdx == matches[i].trainIdx)){

                            good_matches.push_back(matches[i]);
                        }
                    }
                }


                res_y.at<double>(im_it,it) = double(good_matches.size())/double(desc_result_ground_truth.rows);

            }

        }
        //        std::cout<<"Got to load the data"<<std::endl;
        Mat av_y, av_time;
        cv::reduce(res_y,av_y,0,CV_REDUCE_AVG);
        cv::reduce(time_1,av_time,0,CV_REDUCE_AVG);
        //        double descriptor_count_time = double(av_time.at<float>(0,0));

                std::cout<<"Final average vector: "<<av_y<<std::endl;
        //        std::cout<<"Final average descriptor time: "<<descriptor_count_time<<std::endl;

        std::vector<double> final_evaluation_result;
        double results_average = 0;
        for (int av_j=0; av_j<number_of_evaluation_case; ++av_j){

            final_evaluation_result.push_back(av_y.at<double>(0,av_j));
            results_average+=av_y.at<double>(0,av_j);

        }

        results_average = 1.0 - (results_average/number_of_evaluation_case);
        score_results.push_back(results_average*weight[evaluation_type]);
        evaluatedDescriptor->invariantScores.push_back(results_average*weight[evaluation_type]);

        std::cout<<"Results "<<evaluation_type<<" : "<<results_average<<std::endl;

        if(evaluation_type==LIGHT_CH_TR){

            lightChangeResults.push_back(final_evaluation_result);

            //            evaluatedDescriptor->lightChange=final_evaluation_result;
        }
        else if(evaluation_type==LIGHT_COND_TR){

            lightCondition1Results.push_back(final_evaluation_result);
            //            evaluatedDescriptor->lightCondition1=final_evaluation_result;
        }
        else if(evaluation_type==LIGHT_COND_TR2){

            lightCondition2Results.push_back(final_evaluation_result);
            //            evaluatedDescriptor->lightCondition2=final_evaluation_result;
        }
        else if(evaluation_type==JPEG_COMPRESSION){

            jpegCompressionResults.push_back(final_evaluation_result);
            //            evaluatedDescriptor->jpegCompression=final_evaluation_result;
        }
        else if(evaluation_type==ROTATION_TR){

            rotationResults.push_back(final_evaluation_result);
            //            evaluatedDescriptor->rotation=final_evaluation_result;
        }
        else if(evaluation_type==BLUR_TR)
        {
            blurResults.push_back(final_evaluation_result);
            times.push_back(double(av_time.at<float>(0,0)));
            //            evaluatedDescriptor->blur=final_evaluation_result;

        }else if(evaluation_type==AFFINE_TR){

            affineResults.push_back(final_evaluation_result);
            //            evaluatedDescriptor->affine=final_evaluation_result;

        }else if(evaluation_type==SIZE_TR){

            resizeResults.push_back(final_evaluation_result);
            //            evaluatedDescriptor->resize=final_evaluation_result;
        }
    }

    std::cout<<"Finished Transform Evaluation with "<<getDescriptorName(desc_round)<<" descriptor."<<std::endl;
}

double PostGeneticBundleEvaluation::averageScore(){

    double score = 0.0;
    double sum = 0.0;

    for(uint sc_it=0;sc_it<score_results.size();++sc_it){

        sum+=score_results[sc_it];
    }

    score = sum / (double) score_results.size();


    return score;
}

void PostGeneticBundleEvaluation::oxfordEvaluation(){

    std::vector<Images> evaluation_imgs;

    //Set training image database
    Database ox_im_db;

    ox_im_db.setDataBase(OXFORD_DB,usedComputer,TEST,10);

    for(int tim_it=0;tim_it<ox_im_db.size;++tim_it){
        Images one_im;
        one_im.getImagesFromDatabase(ox_im_db,tim_it);
        evaluation_imgs.push_back(one_im);
    }


    string hom_root = ox_im_db.root_folder + ox_im_db.homography_folder_name;

    //    std::cout<<"Matrix: "<<homography_list.at(5)<<std::endl;

    vector<Mat> hom_mtx = loadHomography(hom_root, ox_im_db.size, ox_im_db.homography_list);

    testHomographyMatrix(hom_mtx,evaluation_imgs);

}


int PostGeneticBundleEvaluation::readNumbers( const string & s, vector <double> & v ) {
    istringstream is( s );
    double n;
    while( is >> n ) {
        v.push_back( n );
    }
    return v.size();
}

std::vector<cv::Mat> PostGeneticBundleEvaluation::loadHomography(string folderRoute, int _image_size, const std::vector<string> _hom_list){

    int hom_size;
    std::vector<Mat> hom_mat_vec;

    hom_size = _hom_list.size();


    int mat_count = 0;

    for (int b_im_it=0;b_im_it<_image_size;++b_im_it){

        for (int c_im_it=0;c_im_it<_image_size;++c_im_it){

            if(b_im_it==c_im_it) continue;


            Mat hom_mat(3,3,CV_64FC1,cv::Scalar(0));
            int rows =0;
            int cols =0;

            string file_route = folderRoute + _hom_list[mat_count];

            //            std::cout<<"File route: "<<file_route<<std::endl;

            importMatrixFromFile(file_route.c_str(),hom_mat,rows,cols);

            //            std::cout<<"Homography matrix at base image "<<b_im_it+1<<" coompared with image "<<c_im_it+1<<"\n"<<hom_mat<<"\n"<<std::endl;

            hom_mat_vec.push_back(hom_mat);

            mat_count++;

        }

    }

    return hom_mat_vec;
}

void PostGeneticBundleEvaluation::importMatrixFromFile(const char* filename_X, Mat _hom_mat, int& rows, int& cols){

    ifstream file_X;
    string line;
    vector <double> v;

    file_X.open(filename_X);
    if (file_X.is_open())
    {
        int i=1;
        getline(file_X, line);

        cols =readNumbers( line, v );
        //        cout << "cols:" << cols << endl;


        while (!file_X.eof()){
            getline(file_X, line);
            if (line.empty()){
                break;
            }else{
                readNumbers( line, v );
            }

            i++;
        }

        rows=i;
        //        cout << "rows :" << rows << endl;

        file_X.close();
    }
    else{
        cout << "file open failed";
    }

    //    cout << "v:" << endl;
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            double value = v[i*cols+j];
            //            cout << value << "\t" ;

            _hom_mat.at<double>(i,j)=value;
        }
        //        cout << endl;
    }
}

void PostGeneticBundleEvaluation::testHomographyMatrix(vector<Mat>& homographies, vector<Images>& images){


    int _image_size = images.size();
    int mat_count = 0;

    for (int b_im_it=0;b_im_it<_image_size;++b_im_it){

        Mat src_im = images[b_im_it].col_im;

        float in_w =0.0, in_h = 0.0;

        in_w = (float) src_im.cols;
        in_h = (float) src_im.rows;

        vector<Point2f> object_points,dst_points;

        object_points.push_back(Point2f (0.0f,0.0f));
        object_points.push_back(Point2f (in_w-1,0.0f));
        object_points.push_back(Point2f (in_w-1,in_h-1));
        object_points.push_back(Point2f (0.0f,in_h-1));

        Mat dst;

        for (int c_im_it=0;c_im_it<_image_size;++c_im_it){

            if(b_im_it==c_im_it) continue;

            Mat match_im = images[c_im_it].col_im;

            cv::perspectiveTransform(Mat(object_points),dst,homographies[mat_count]);

            //            std::cout<<"Dest mtrx: "<<dst<<std::endl;

            vector<Point>dst_points;

            for (int pt_it=0;pt_it<dst.rows;++pt_it){

                dst_points.push_back(Point((int)dst.at<float>(pt_it,0),(int)dst.at<float>(pt_it,1)));

            }

            const cv::Point *pts = (const cv::Point*) Mat(dst_points).data;
            int npts = Mat(dst_points).rows;

            polylines(match_im, &pts,&npts, 1,
                      true, 			// draw closed contour (i.e. joint end to start)
                      Scalar(0,255,0),// colour RGB ordering (here = green)
                      3, 		        // line thickness
                      CV_AA, 0);

            cv::imshow("Original image",src_im);
            cv::imshow("Transformed image",match_im);

            cv::waitKey();

            mat_count++;
        }
    }
}

void PostGeneticBundleEvaluation::plotResults(){



    for (int p_i=0;p_i<8;++p_i){

        PostDialog w;
        BarDialog wb;

        if(p_i==ROTATION_TR){
            w.cases_x = {0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360};
            w.sift_plot_vector = rotationResults[SIFT_D];
            w.surf_plot_vector = rotationResults[SURF_D];
            w.orb_plot_vector = rotationResults[ORB_D];
            w.brief_plot_vector =  rotationResults[BRIEF_D];
            w.brisk_plot_vector =  rotationResults[BRISK_D];
            w.freak_plot_vector =  rotationResults[FREAK_D];
            w.latch_plot_vector =  rotationResults[LATCH_D];
            w.edd_plot_vector = rotationResults[M_PROJECT_D];
            w.project_plot_vector = rotationResults[D_PROJECT_D];
            w.plot(ROTATION_TR);
        }else  if(p_i==AFFINE_TR){
            w.cases_x = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
            w.sift_plot_vector = affineResults[SIFT_D];
            w.surf_plot_vector = affineResults[SURF_D];
            w.orb_plot_vector = affineResults[ORB_D];
            w.brief_plot_vector =  affineResults[BRIEF_D];
            w.brisk_plot_vector =  affineResults[BRISK_D];
            w.freak_plot_vector =  affineResults[FREAK_D];
            w.latch_plot_vector =  affineResults[LATCH_D];
            w.edd_plot_vector = affineResults[M_PROJECT_D];
            w.project_plot_vector = affineResults[D_PROJECT_D];
            w.plot(AFFINE_TR);
        }else  if(p_i==SIZE_TR){
            w.cases_x = {0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5};
            w.sift_plot_vector = resizeResults[SIFT_D];
            w.surf_plot_vector = resizeResults[SURF_D];
            w.orb_plot_vector = resizeResults[ORB_D];
            w.brief_plot_vector =  resizeResults[BRIEF_D];
            w.brisk_plot_vector =  resizeResults[BRISK_D];
            w.freak_plot_vector =  resizeResults[FREAK_D];
            w.latch_plot_vector =  resizeResults[LATCH_D];
            w.edd_plot_vector = resizeResults[M_PROJECT_D];
            w.project_plot_vector = resizeResults[D_PROJECT_D];
            w.plot(SIZE_TR);
        }else  if(p_i==BLUR_TR){
            w.cases_x = {1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0};
            w.sift_plot_vector = blurResults[SIFT_D];
            w.surf_plot_vector = blurResults[SURF_D];
            w.orb_plot_vector = blurResults[ORB_D];
            w.brief_plot_vector =  blurResults[BRIEF_D];
            w.brisk_plot_vector =  blurResults[BRISK_D];
            w.freak_plot_vector =  blurResults[FREAK_D];
            w.latch_plot_vector =  blurResults[LATCH_D];
            w.edd_plot_vector = blurResults[M_PROJECT_D];
            w.project_plot_vector = blurResults[D_PROJECT_D];

            wb.time_plot_vector.push_back(times[SIFT_D]);
            wb.time_plot_vector.push_back(times[SURF_D]);
            wb.time_plot_vector.push_back(times[ORB_D]);
            wb.time_plot_vector.push_back(times[BRIEF_D]);
            wb.time_plot_vector.push_back(times[BRISK_D]);
            wb.time_plot_vector.push_back(times[FREAK_D]);
            wb.time_plot_vector.push_back(times[LATCH_D]);
            wb.time_plot_vector.push_back(times[M_PROJECT_D]);
            wb.time_plot_vector.push_back(times[D_PROJECT_D]);

            std::cout<<"Times: SIFT: "<<times[SIFT_D]
            <<", SURF: "<<times[SURF_D]
            <<", ORB: "<<times[ORB_D]
            <<", BRIEF: "<<times[BRIEF_D]
            <<", BRISK: "<<times[BRISK_D]
            <<", FREAK: "<<times[FREAK_D]
            <<", LATCH: "<<times[LATCH_D]
            <<", EDD: "<<times[M_PROJECT_D]
            <<", MODULAR: "<<times[D_PROJECT_D]
            <<std::endl;

            w.plot(BLUR_TR);
            wb.plotBar();
            wb.show();

        }else  if(p_i==LIGHT_CH_TR){
            w.cases_x = {0,1,2,3,4};
            w.sift_plot_vector = lightChangeResults[SIFT_D];
            w.surf_plot_vector = lightChangeResults[SURF_D];
            w.orb_plot_vector = lightChangeResults[ORB_D];
            w.brief_plot_vector =  lightChangeResults[BRIEF_D];
            w.brisk_plot_vector =  lightChangeResults[BRISK_D];
            w.freak_plot_vector =  lightChangeResults[FREAK_D];
            w.latch_plot_vector =  lightChangeResults[LATCH_D];
            w.edd_plot_vector = lightChangeResults[M_PROJECT_D];
            w.project_plot_vector = lightChangeResults[D_PROJECT_D];
            w.plot(LIGHT_CH_TR);
        }else  if(p_i==LIGHT_COND_TR){
            w.cases_x = {0,1,2,3,4};
            w.sift_plot_vector = lightCondition1Results[SIFT_D];
            w.surf_plot_vector = lightCondition1Results[SURF_D];
            w.orb_plot_vector = lightCondition1Results[ORB_D];
            w.brief_plot_vector =  lightCondition1Results[BRIEF_D];
            w.brisk_plot_vector =  lightCondition1Results[BRISK_D];
            w.freak_plot_vector =  lightCondition1Results[FREAK_D];
            w.latch_plot_vector =  lightCondition1Results[LATCH_D];
            w.edd_plot_vector = lightCondition1Results[M_PROJECT_D];
            w.project_plot_vector = lightCondition1Results[D_PROJECT_D];
            w.plot(LIGHT_COND_TR);
        }else  if(p_i==JPEG_COMPRESSION){
            w.cases_x = {0,1,2,3,4};
            w.sift_plot_vector = jpegCompressionResults[SIFT_D];
            w.surf_plot_vector = jpegCompressionResults[SURF_D];
            w.orb_plot_vector = jpegCompressionResults[ORB_D];
            w.brief_plot_vector =  jpegCompressionResults[BRIEF_D];
            w.brisk_plot_vector =  jpegCompressionResults[BRISK_D];
            w.freak_plot_vector =  jpegCompressionResults[FREAK_D];
            w.latch_plot_vector =  jpegCompressionResults[LATCH_D];
            w.edd_plot_vector = jpegCompressionResults[M_PROJECT_D];
            w.project_plot_vector = jpegCompressionResults[D_PROJECT_D];
            w.plot(JPEG_COMPRESSION);
        }else  if(p_i==LIGHT_COND_TR2){
            w.cases_x = {0,1,2,3,4,5,6,7,8};
            w.sift_plot_vector = lightCondition2Results[SIFT_D];
            w.surf_plot_vector = lightCondition2Results[SURF_D];
            w.orb_plot_vector = lightCondition2Results[ORB_D];
            w.brief_plot_vector =  lightCondition2Results[BRIEF_D];
            w.brisk_plot_vector =  lightCondition2Results[BRISK_D];
            w.freak_plot_vector =  lightCondition2Results[FREAK_D];
            w.latch_plot_vector =  lightCondition2Results[LATCH_D];
            w.edd_plot_vector = lightCondition2Results[M_PROJECT_D];
            w.project_plot_vector = lightCondition2Results[D_PROJECT_D];
            w.plot(LIGHT_COND_TR2);
        }
        w.show();
        //        if (evaluation_type == BLUR){
        //            w.project_plot_vector = {{1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0},
        //                                     {100.0, 88.6364, 84.899, 74.9495, 64.2424, 57.7273, 49.7475, 42.6263, 37.6263, 32.9798}};
        //        }else if (evaluation_type == ROTATE){
        //            w.project_plot_vector = {{0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360},
        //                                     {100.0, 44.9405, 29.5238, 18.9286, 12.7381, 10.0595, 7.61905, 5.59524, 4.7619, 4.58333, 4.22619, 3.80952, 4.16667, 3.80952, 3.33333, 4.28571, 4.88095, 5.83333, 8.63095, 9.16667, 10.4762, 19.3452, 33.8095, 63.2738, 100.0}};
        //        }else if (evaluation_type == AFFINE){
        //             w.project_plot_vector = {{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0},
        //                                      {83.3929, 67.0238, 46.6071, 31.8452, 17.9762, 67.5595, 33.3929, 11.9048, 1.30952, 9.7619}};
        //        }else if (evaluation_type == SIZE){
        //            w.project_plot_vector = {{0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.5},
        //                                     {15.6548, 17.2024, 48.1548, 78.8095, 100.0, 81.8452, 58.9881, 27.6786, 5.53571}};
        //       }

        //    wb.time_plot_vector.push_back(seg_proc5.descriptor_count_time);
        //!=================================================================
    }
}

void PostGeneticBundleEvaluation::colorAndSaveClassResults(vector<KeyPoint> _keypoints, cv::Mat _image, cv::Mat _results, cv::Mat _entrophy, int _desc_round, int _im_turn){

    Mat col_image, ent_image,rgb;
    double alpha = 0.5;
    double beta;

    cv::cvtColor(_image,rgb,CV_RGB2BGR);
    col_image = rgb.clone();
    ent_image = rgb.clone();


    string descriptor_name = getDescriptorName(_desc_round);

    //Colouring the keypoint to white if it is Window or Door
    for (int kp=0; kp<(int)_keypoints.size(); ++kp){
        //at each keypoint colour

        float res = _results.at<float>(kp,0);
        //and create a red circle if entropy high, white if it is low
        float ent_col =_entrophy.at<float>(kp,0)*255;
        //std::cout<<"Results at: "<<kp<<"  "<<res<<std::endl;
        cv::circle(ent_image, cv::Point(_keypoints[kp].pt.x, _keypoints[kp].pt.y), 1, cv::Scalar(255-ent_col,255-ent_col,255),2);

        if(res==0){
            cv::circle( col_image, cv::Point(_keypoints[kp].pt.x, _keypoints[kp].pt.y), 1, cv::Scalar(0,255, 0) );
        }else if(res==1){
            cv::circle( col_image, cv::Point(_keypoints[kp].pt.x, _keypoints[kp].pt.y), 1, cv::Scalar(0,0, 255) );
        }else if (res==2){
            cv::circle( col_image, cv::Point(_keypoints[kp].pt.x, _keypoints[kp].pt.y), 1, cv::Scalar(255,0, 0) );
        }else if (res==3){
            cv::circle( col_image, cv::Point(_keypoints[kp].pt.x, _keypoints[kp].pt.y), 1, cv::Scalar(0,255,255) );
        }
    }


    string folderName = "../results/images";
    char im_name[200];
    char ent_im_name[200];
    const char *dn = descriptor_name.c_str();
    sprintf(im_name, "%s_descriptor_result_%d.png",dn, _im_turn);
    sprintf(ent_im_name, "%s_descriptor_entrophy_result_%d.png",dn, _im_turn);
    string im_fileName = folderName+"/"+string(im_name);
    string ent_im_fileName = folderName+"/"+string(ent_im_name);

    beta = ( 1.0 - alpha );
    addWeighted( rgb, alpha, col_image, beta, 0.0, col_image);
    addWeighted( rgb, alpha, ent_image, beta, 0.0, ent_image);


    imwrite ( im_fileName, col_image );
    imwrite ( ent_im_fileName, ent_image);



}


string PostGeneticBundleEvaluation::getDescriptorName(int _desc_round) {

    string descriptor_name = "";


    switch (_desc_round) {
        case SIFT_D:
            descriptor_name = "SIFT";
            break;
        case SURF_D:
            descriptor_name = "SURF";
            break;
        case ORB_D:
            descriptor_name = "ORB";
            break;
        case BRIEF_D:
            descriptor_name = "BRIEF";
            break;
        case BRISK_D:
            descriptor_name = "BRISK";
            break;
        case FREAK_D:
            descriptor_name = "FREAK";
            break;
        case LATCH_D:
            descriptor_name = "LATCH";
            break;
        case M_PROJECT_D:
            descriptor_name = "EDD";
            break;
        case D_PROJECT_D:
            descriptor_name = "Modular";
            break;
        default: descriptor_name = "Descriptor hasn't been set!";
    }

    return descriptor_name;
}
