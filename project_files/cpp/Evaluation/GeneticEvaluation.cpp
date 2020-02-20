#include "Evaluation/GeneticEvaluation.h"

GeneticEvaluation::GeneticEvaluation()
{
    descriptor_global_score=0.0;
}

GeneticEvaluation::GeneticEvaluation(std::shared_ptr<MDADescriptor> _descriptor, const int& _database, int _eval_im_number)
{
    descriptor_global_score=0.0;
    evaluatedDescriptor = _descriptor;
    assembledDescriptor = _descriptor->getModuleList();

    //weighting the results. Order of results:(0-8)Evaluation results, (9)Extraction Time, (10-)Classification results
    weight = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    usedDatabase = _database;
    number_of_evaluation_images = _eval_im_number;
    mark_keypoints = false;
    matcher_type = BRUTE_FORCE; //It can be FLANN or BRUTE_FORCE
    comparation_method=DESCRIPTOR_INDEX; // COMPARATION METHOD: "DOT_PRODUCT", "DESCRIPTOR_INDEX"
    do_training_test = true;
    do_transformation_test = true;
}

GeneticEvaluation::~GeneticEvaluation()
{

}

void GeneticEvaluation::evaluate(){

    //oxfordEvaluation();

//    transformEvaluation();

if(do_training_test && evaluatedDescriptor->getModuleListSize()>0){
    trainingEvaluation();
} if(!do_training_test){

  evaluatedDescriptor->timeScores.push_back(0.0);
  for(int si=0;si<5;++si){
      double _score = 0.0;
      score_results.push_back(_score);
      evaluatedDescriptor->segmentationScores.push_back(_score);

  }

} if(do_transformation_test && evaluatedDescriptor->getModuleListSize()>0){
    transformEvaluation2();
} if(!do_transformation_test){
  for(int evaluation_type=0;evaluation_type<8;++evaluation_type){
    evaluatedDescriptor->invariantScores.push_back(0);
  }
}


}

void GeneticEvaluation::trainingEvaluation(){

    cv::Mat total_desc,total_label;
    int loop_start,loop_finish;
    int patch_radius = evaluatedDescriptor->getPatchRadius();

    //Set training image database
    Database im_db;
    std::cout<<"Used computer: "<<usedComputer<<std::endl;
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

        desc_result = modularDescriptor.calculate(im,train_kp);
        label_result = labelExtractor.labelingImage(im,train_kp);
        total_desc.push_back(desc_result);
        total_label.push_back(label_result);

    }

    std::cout<<"Extracted descriptor matrix size: "<<total_desc.cols<<", "<<total_label.rows<<std::endl;

    //INSTANTIATE RandomTree for train and test
    //    RandomForest* rtree = new RandomForest();
    std::unique_ptr<RandomForest> rforest(new RandomForest());

    std::cout<<"Start tree tarining!"<<std::endl;

    int rf_depth = evaluatedDescriptor->getTreeDepth();
    int rf_size = evaluatedDescriptor->getTreeNumbers();
    int rf_split = evaluatedDescriptor->getSplitNumber();

    //TRAINING the tree
    rforest->train(total_desc,total_label,D_PROJECT_D,rf_depth,rf_size,rf_split);
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
    for(int tst=t_loop_start;tst<t_loop_finish;++tst)
    {
        cv::Mat measured_times(1,2,CV_64FC1);
        Mat test_descriptors, test_labels;

        Images test_im;
        test_im.getImagesFromDatabase(im_db,tst);

        if(image_key_points.theImageIsDifferent(test_im.col_im)){
            image_key_points.calculateCoordinates(test_im);
        }

        descriptor_ext_start = std::clock();

        std::cout<<"Got testing images!"<<std::endl;

        //Calculate the test descriptor and labels
        vector<KeyPoint> test_kp = image_key_points.getKeyPoints();
        test_descriptors = modularDescriptor.calculate(test_im,test_kp);
        test_labels = labelExtractor.labelingImage(test_im,test_kp);

        std::cout<<"Got testing labels and descriptors!"<<std::endl;

        descriptor_ext_duration = ( std::clock() - descriptor_ext_start ) / (double) CLOCKS_PER_SEC;
        measured_times.at<double>(0,0)=descriptor_ext_duration;

        //Initialize image matrices
        cv::Mat results(test_descriptors.rows,1,CV_32FC1);
        cv::Mat entropy(test_descriptors.rows,1,CV_32FC1);
        cv::Mat res_percent(1,NUMBER_OF_CLASSES+1,CV_32FC1);

        std::cout<<"The results of test at Image "<<tst<<":"<<std::endl;
        rforest->test( test_descriptors,test_labels, results, entropy, res_percent, D_PROJECT_D);

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

    cv::reduce(all_res_percent,av_percent,0,CV_REDUCE_AVG,-1);
    cv::reduce(all_measured_times,av_times,0,CV_REDUCE_AVG,-1);

    std::cout<<"The tree test average result: "<<av_percent<<std::endl;

    evaluatedDescriptor->segmentationResults = av_percent;
    evaluatedDescriptor->calculationTimesResult = av_times;

    av_percent = av_percent*0.01;

    av_percent.convertTo(percent_score,CV_64F);
    av_times.convertTo(time_score,CV_64F);


    //Adds the weighted time score and the weighted classification score to the results vector
    double _time_score = time_score.at<double>(0,0)*weight.at(8);
    score_results.push_back(_time_score);
    evaluatedDescriptor->timeScores.push_back(_time_score);
    std::cout<<"Time Score: "<<time_score.at<double>(0,0)*weight.at(8)<<std::endl;
    for(int si=0;si<percent_score.cols;++si){
        double _score = 1.0-(percent_score.at<double>(0,si)*weight.at(si+9));
        score_results.push_back(_score);
        evaluatedDescriptor->segmentationScores.push_back(_score);

    }
    av_percent.release();
    av_times.release();
    time_score.release();
    percent_score.release();

    std::cout<<"Finished training and testing the current descriptor with Random Trees"<<std::endl;

}


void GeneticEvaluation::transformEvaluation(){

    std::vector<Images> evaluation_imgs;

    //    std::cout<<"Image loading starts"<<std::endl;
    //! Get test images
    Database eval_im_db;
    eval_im_db.setDataBase(usedDatabase, usedComputer,TRAIN);

    for(int tim_it=0;tim_it<number_of_evaluation_images;++tim_it){
        Images one_im;
        one_im.getImagesFromDatabase(eval_im_db,tim_it);
        evaluation_imgs.push_back(one_im);
    }

    //    std::cout<<"Images are loaded"<<std::endl;


    EvaluationValues ev_values;

    //Evaluation_type(enumeration value):
    // 0. Rotation(4)
    // 1. Affine Transformation(6)
    // 2. Resize(5)
    // 3. Blur(3)
    // 4. Intensity Change(1)
    // 5. Intensity Shift(2)
    // 6. Intesity Temperature(0)

    std::vector<string> eval_name = {"Rotation", "Affine Transformation", "Resize", "Blur", "Intensity Change", "Intensity Shift", "Intesity Temperature"};
    std::vector<int> current_eval_max = {ev_values.rotations_vec_length(),ev_values.affine_vec_length(),ev_values.sizes_vec_length(),ev_values.kernels_vec_length(),
                                         ev_values.alphas_vec_length(),ev_values.betas_vec_length(),ev_values.alphaTriplets_vec_length()};


    //run all the evaluation cases---> running parallel with pragma
//#pragma omp parallel for
    for(int evaluation_type=0;evaluation_type<7;++evaluation_type){

        std::cout<<eval_name[evaluation_type]<<" evaluation starts!"<<std::endl;

        int number_of_images, number_of_evaluation_case;

        clock_t clk1, clk2, clk3;

        int start_iteration =0;
        int end_iteration = current_eval_max[evaluation_type];


        //setup values for the result matrix
        number_of_images = evaluation_imgs.size();
        number_of_evaluation_case = end_iteration;

        Mat res_y = Mat::zeros(number_of_images,end_iteration,CV_64FC1);
        Mat time_1 = Mat::zeros(1,1,CV_32FC1);

        for(int im_it=0;im_it<number_of_images;++im_it){

            //            std::cout<<"Image "<<im_it<<" is under process!"<<std::endl;

            Images imgs_1, imgs_2;
            Mat transf_mat, im_for_desc1, im_for_desc2;

            imgs_1 = evaluation_imgs.at(im_it);

            for(int it=start_iteration; it<end_iteration;++it){

                //If evaluation type is Rotation, Affine Transformation or Resize creates transformation matrix
                if (evaluation_type<3){

                    if(evaluation_type==0){

                        imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(evaluation_imgs.at(im_it),ROTATE);

                        transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5),ev_values.rotationAt(it), 1 );

                        imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);

                    }else if(evaluation_type==1){

                        transf_mat = ImageProcessing::getAffineTransformMat(imgs_1.col_im.rows,imgs_1.col_im.cols,ev_values.affineTripletAt(it));
                        //resize image box for the transformed image
                        imgs_2 = imgs_1.affineTransformImagesA(transf_mat,ev_values.affineTripletAt(it));

                    }else if(evaluation_type==2){

                        imgs_1 = ImageProcessing::enlargeImageForRotationAndSize(evaluation_imgs.at(im_it),SIZE);

                        transf_mat = cv::getRotationMatrix2D( Point2f(imgs_1.col_im.cols*0.5,imgs_1.col_im.rows*0.5), 0, ev_values.sizeAt(it));

                        imgs_2 = imgs_1.affineTransformImagesRS(transf_mat);
                    }

                    //transform the evaluation image with the specific transformation matrix
                    //                    imgs_2 = imgs_1.affineTransformImages(transf_mat);


                }else if(evaluation_type==3){


                    cv::GaussianBlur(imgs_1.col_im,imgs_2.col_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                    cv::GaussianBlur(imgs_1.grey_im,imgs_2.grey_im,Size(ev_values.kernelSizeAt(it),ev_values.kernelSizeAt(it)),0,0,BORDER_CONSTANT);
                    imgs_1.mask_im = imgs_2.mask_im;

                }else if(evaluation_type==4){

                    imgs_2.col_im = ImageProcessing::lightIntensityChangeAndShift(imgs_1.col_im, ev_values.alphaAt(it), 0);
                    cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

                }else if(evaluation_type==5){

                    imgs_2.col_im = ImageProcessing::lightIntensityChangeAndShift(imgs_1.col_im,1, ev_values.betaAt(it));
                    cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

                }else if(evaluation_type==6){

                    imgs_2.col_im = ImageProcessing::colorTemperatureChange(imgs_1.col_im,ev_values.alphaTripletAt(it), 0);
                    cv::cvtColor(imgs_2.col_im,imgs_2.grey_im,CV_RGB2GRAY);

                }


                //!-- Step 1: Detect the keypoints
                //!========================================= KEYPOINT=====================================================

                int patch_radius = evaluatedDescriptor->getPatchRadius();
                int margin=0;

                if(evaluation_type<3){
                    margin = patch_radius*2+1;
                }
                //

                //                imshow("Blured Image",imgs_2.col_im);
                //                imshow("Normal Image",imgs_1.col_im);

                //                waitKey();

                //Instantiate keypoint class

                ImageKeypoints evaluation_keypoints(PATCH_DIST_KP,imgs_1.col_im,margin,patch_radius);

                evaluation_keypoints.calculateCoordinates(imgs_1);

                if(evaluation_type<3){
                    evaluation_keypoints.transformAllKeypoints(transf_mat);
                }

                if(mark_keypoints){
                    //! to color the keypoints on the transformed image
                    for (uint i=0; i<evaluation_keypoints.keypoints.size();++i){

                        KeyPoint kp_1 = evaluation_keypoints.keypoints[i];
                        KeyPoint kp_2 = evaluation_keypoints.transformedKeypoints[i];

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

                clk1 = clock();
                desc_result_ground_truth = modularDescriptor.calculate(imgs_1,evaluation_keypoints.keypoints);
                clk2 = clock();
                desc_result_transformed = modularDescriptor.calculate(imgs_2,evaluation_keypoints.transformedKeypoints);
                clk3 = clock();


                time_1.push_back( (float)(clk2-clk1)/CLOCKS_PER_SEC);
                time_1.push_back( (float)(clk3-clk2)/CLOCKS_PER_SEC);
                //-- Step 3: Matching descriptor vectors using FLANN matcher


                std::vector< DMatch > matches;
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
                    matcher.match( desc_result_ground_truth, desc_result_transformed, matches );
                    //                matcher.match( comp_desc1, comp_desc2, matches );

                }
                else if(matcher_type==BRUTE_FORCE){

                    BFMatcher matcher(NORM_L2, true);
                    matcher.knnMatch( desc_result_ground_truth, desc_result_transformed, knn_matches,1);
                    //                matcher.knnMatch( comp_desc1, comp_desc2, knn_matches,1);

                }

                double max_dist = 0; double min_dist = 1000000000;

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


                //            printf("-- Max dist : %f \n", max_dist );
                //            printf("-- Min dist : %f \n", min_dist );

                //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
                //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
                //-- small)
                //-- PS.- radiusMatch can also be used here.

                std::vector< DMatch > good_matches;
                int desc_size = desc_result_ground_truth.rows;


                //            std::cout<<"Vec Degree: "<<vec_degree.rows<<" Descriptor Size: "<<desc_size<<std::endl;
                for( int i = 0; i < desc_result_ground_truth.rows; i++ )
                {
                    //                if( knn_matches[i][0].distance <=  0.02 /*max(min_dist*5, 0.02)*/) {
                    //                    good_matches.push_back(knn_matches[i][0]);
                    //                }

                    if(comparation_method==DESCRIPTOR_INDEX){
                        if((matcher_type==BRUTE_FORCE) && (knn_matches[i][0].queryIdx == knn_matches[i][0].trainIdx)){
                            good_matches.push_back(knn_matches[i][0]);

                        }else if((matcher_type==FLANN)&&(matches[i].queryIdx == matches[i].trainIdx)){

                            good_matches.push_back(matches[i]);
                        }
                    }else if(comparation_method==DOT_PRODUCT){


                        bool empty = false;
                        int q_idx=0,t_idx=0;
                        if(matcher_type==BRUTE_FORCE){
                            q_idx = knn_matches[i][0].queryIdx;
                            t_idx = knn_matches[i][0].trainIdx;
                            empty = knn_matches[i].empty();
                        }else if(matcher_type==FLANN){
                            q_idx = matches[i].queryIdx;
                            t_idx = matches[i].trainIdx;
                        }
                        double dot_desc = 0;

                        if(max_dist>DBL_EPSILON && !empty){

                            if(q_idx>=0 && q_idx<desc_size && t_idx>=0 && t_idx<desc_size){
                                Mat in_dot, out_dot;

                                cv::normalize(desc_result_ground_truth.row(q_idx),in_dot);
                                cv::normalize(desc_result_transformed.row(q_idx),out_dot);

                                dot_desc = acos(in_dot.dot(out_dot))*180/PI;
                                //                        vec_degree.push_back(dot_desc);
                                //                        std::cout<<"In dot: "<<in_dot<<std::endl;
                                //                        std::cout<<"Descriptor dot: "<<dot_desc<<" distance: "<<knn_matches[i][0].distance<<std::endl;
                                in_dot.release();out_dot.release();
                            }
                        }else if (max_dist>=0 && max_dist<DBL_EPSILON){

                            dot_desc = 0.0;

                        }else{
                            dot_desc = 15.0;
                            //                    std::cout<<"Descriptor distance: "<<knn_matches[i][0].distance<<" index "<<knn_matches[i][0].imgIdx<<std::endl;

                        }

                        if (dot_desc<5){
                            if(matcher_type==BRUTE_FORCE){
                                good_matches.push_back(knn_matches[i][0]);
                            }else if(matcher_type==FLANN){
                                good_matches.push_back(matches[i]);
                            }


                        }

                    }
                }
                //        printf("Good Matches: %f, Decriptor number: %f, Percent of good matches: %f%% \n", double(good_matches.size()),double(descriptors_1.descriptors.rows),(double(good_matches.size())/double(descriptors_1.descriptors.rows)) );
                //            std::cout<<"At round no. "<<it<<" the Good Matches: "<<double(good_matches.size())/double(descriptors_1.descriptors.rows)<<std::endl;
                //            std::cout<<"Image no.: "<<im_it<<" at round: "<<it<<" the min distance is: "<< min_dist<<" and max dist: "<<max_dist<<std::endl;

                //!        -- Draw only "good" matches======================================
                //            if(evaluation_type!=ROTATION){
                //                if(it==(end_iteration-1)){
                //                    drawMatches( imgs_1.col_im, evaluation_keypoints.keypoints, imgs_2.col_im, evaluation_keypoints.transformedKeypoints,
                //                                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                //                                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                //                    altered_im = imgs_2.col_im;
                //                }
                //            }

                res_y.at<double>(im_it,it) = double(good_matches.size())/double(desc_result_ground_truth.rows);

            }

        }

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

        if(evaluation_type==INTENSITY_SHIFT){

            intensityShiftResults=final_evaluation_result;
            evaluatedDescriptor->intensityShift=final_evaluation_result;
        }
        else if(evaluation_type==INTENSITY_CHANGE){

            intensityChangeResults=final_evaluation_result;
            evaluatedDescriptor->intensityChange=final_evaluation_result;
        }
        else if(evaluation_type==INTENSITY_TEMP){

            colorTemperatureResults=final_evaluation_result;
            evaluatedDescriptor->colorTemperature=final_evaluation_result;
        }
        else if(evaluation_type==ROTATE){

            rotationResults=final_evaluation_result;
            evaluatedDescriptor->rotation=final_evaluation_result;
        }
        else if(evaluation_type==BLUR)
        {
            blurResults=final_evaluation_result;
            evaluatedDescriptor->blur=final_evaluation_result;

        }else if(evaluation_type==AFFINE){

            affineResults=final_evaluation_result;
            evaluatedDescriptor->affine=final_evaluation_result;

        }else if(evaluation_type==SIZE){

            resizeResults=final_evaluation_result;
            evaluatedDescriptor->resize=final_evaluation_result;
        }
    }
}

void GeneticEvaluation::transformEvaluation2(){

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
//                start_iteration = 1;
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
                        if(evaluation_type==ROTATION_TR ||evaluation_type==AFFINE_TR ||evaluation_type==SIZE_TR || evaluation_type==LIGHT_CH_TR || evaluation_type==LIGHT_COND_TR || evaluation_type==JPEG_COMPRESSION){
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

                clk1 = clock();
                desc_result_ground_truth = modularDescriptor.calculate(imgs_1,evaluation_keypoints.keypoints);
                clk2 = clock();
                desc_result_transformed = modularDescriptor.calculate(imgs_2,evaluation_keypoints.transformedKeypoints);
                clk3 = clock();



                time_1.push_back( (float)(clk2-clk1)/CLOCKS_PER_SEC);
                time_1.push_back( (float)(clk3-clk2)/CLOCKS_PER_SEC);
                //-- Step 3: Matching descriptor vectors using FLANN matcher


                std::vector< DMatch > matches;
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
        std::cout<<"Got to load the data"<<std::endl;
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

            lightChangeResults=final_evaluation_result;
            evaluatedDescriptor->lightChange=final_evaluation_result;
        }
        else if(evaluation_type==LIGHT_COND_TR){

            lightCondition1Results=final_evaluation_result;
            evaluatedDescriptor->lightCondition1=final_evaluation_result;
        }
        else if(evaluation_type==LIGHT_COND_TR2){

            lightCondition2Results=final_evaluation_result;
            evaluatedDescriptor->lightCondition2=final_evaluation_result;
        }
        else if(evaluation_type==JPEG_COMPRESSION){

            jpegCompressionResults=final_evaluation_result;
            evaluatedDescriptor->jpegCompression=final_evaluation_result;
        }
        else if(evaluation_type==ROTATION_TR){

            rotationResults=final_evaluation_result;
            evaluatedDescriptor->rotation=final_evaluation_result;
        }
        else if(evaluation_type==BLUR_TR)
        {
            blurResults=final_evaluation_result;
            evaluatedDescriptor->blur=final_evaluation_result;

        }else if(evaluation_type==AFFINE_TR){

            affineResults=final_evaluation_result;
            evaluatedDescriptor->affine=final_evaluation_result;

        }else if(evaluation_type==SIZE_TR){

            resizeResults=final_evaluation_result;
            evaluatedDescriptor->resize=final_evaluation_result;
        }
    }
}

double GeneticEvaluation::averageScore(){

    double score = 0.0;
    double sum = 0.0;

    for(uint sc_it=0;sc_it<score_results.size();++sc_it){

        sum+=score_results[sc_it];
    }

    score = sum / (double) score_results.size();


    return score;
}

void GeneticEvaluation::oxfordEvaluation(){

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


int GeneticEvaluation::readNumbers( const string & s, vector <double> & v ) {
    istringstream is( s );
    double n;
    while( is >> n ) {
        v.push_back( n );
    }
    return v.size();
}

std::vector<cv::Mat> GeneticEvaluation::loadHomography(string folderRoute, int _image_size, const std::vector<string> _hom_list){

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

void GeneticEvaluation::importMatrixFromFile(const char* filename_X, Mat _hom_mat, int& rows, int& cols){

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

void GeneticEvaluation::testHomographyMatrix(vector<Mat>& homographies, vector<Images>& images){


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
