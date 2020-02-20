#include "utils.h"
#include "Evaluation/EvaluationValues.h"
#include <QApplication>
#include "Dataset/ParsingCSV.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

string homeDirectory = "../";
//! If evaluating-->
//! number_of_test_images = 0;
//! number_of_images = 1;

//uint number_of_test_images = 1;
//uint number_of_images = 48;

uint number_of_test_images = 0;
uint number_of_images = 10;


uint descriptor_width = 113;
//uint IM_WIDTH = 640;
//uint IM_HEIGHT = 480;
uint NUMBER_OF_CLASSES = 4;


bool training_masters = false;  //Training different machine learning algorithms
bool evaluation = false; //Simple descriptor evaluation
bool evaluation2 = false; //Post genetic evaluation
bool evaluation3 = true; //Bundle Post genetic evaluation
bool genetic = false; //Genetic Algorithm
bool tilde = false; //Tilde algorithm



//!If tree exist value is true
bool test_tree = true;
bool trained = false;

//!type of macine learning algorithm
bool randomTrees = true;
bool bag_of_words = false;
bool naive_bayes = false;
bool suppor_vector_machine = false;
bool neural_networks = false;


int main()
{

    char *app_name = (char *) "Descriptor Optimizer";
    char *_arg1 = (char *) "arg1";
    char *_arg2 = (char *) "arg2";

    char *argv[] = {app_name, _arg1, _arg2, NULL};
    int argc = sizeof(argv) / sizeof(char*) - 1;

    QApplication a(argc, argv);
    //    QLabel *label = new QLabel("Hello Qt!");
    //    label->show();


    cv::Mat test_descriptors;
    cv::Mat train_descriptors;
    cv::Mat test_labels( 500, 1, CV_32FC1 );
    cv::Mat train_labels( 500, 1, CV_32FC1 );


    //==========================================================
    //DIFFERENT FUNCTION FOR THE descriptorextractor Function
    //
    //
    // TForest type: "TRAIN" or "TEST"
    //
    // Key points: "FAST","RAND", "ALL", "PATCH_DIST"
    //
    // Descriptor types: "RGB", "OPP_SURF", "OPP_SIFT" , "SPECIAL", "SIFT", "SURF"
    //
    //==========================================================

    if(training_masters){

        for(int decriptor_turn=4;decriptor_turn<5;++decriptor_turn){


            if(decriptor_turn==6){

                continue;
            }


            KeyInterestPoint keypoints1;
            int trainPercent = 100;
            //            int numberOfTestImages = 1;
            int patch_radius = 4;



            //            RandomForest rtreeObject;
            //            RandomForest* rtree = &rtreeObject;//new RandomForest();
            std::unique_ptr<RandomForest> rtree(new RandomForest());//new RandomForest();


            BagOfWords bagOfWordsObject;

            //            NaiveBayes bayesObject;
            //            NaiveBayes* bayesClassificator = &bayesObject;
            std::unique_ptr<NaiveBayes> bayesClassificator(new NaiveBayes());

            SupportVectorMachine suppVecMach;

            //            NeuralNetworks annObj;
            //            NeuralNetworks* annClassifier = &annObj;
            std::unique_ptr<NeuralNetworks> annClassifier(new NeuralNetworks());

            if(trained==false){

                //get descriptors and labelsfor training
                //              keypoints1.extractDescriptorsAndLables("TRAIN","PATCH_DIST",M_PROJECT_D,train_descriptors,train_labels,trainig_imgs,25,0,trainig_imgs.size());

                keypoints1.extraction(TRAIN,PATCH_DIST_KP,PROJECT_2_CLASS,decriptor_turn,train_descriptors,train_labels,trainPercent,patch_radius);
                std::cout<<"Got keypoints"<<std::endl;

                std::cout<<"Training started"<<std::endl;


                //train the tree
                double training_duration;
                std::clock_t training_time;
                training_time = std::clock();

                if(randomTrees==true){

                    rtree->train(train_descriptors,train_labels,decriptor_turn);

                }else if(bag_of_words==true){

                    //Calculating the unsupervised Bag of word centers
                    //                    bagOfWordsObject.calculateUnSupervisedVocabulary(train_descriptors,4);

                    //Calculating the supervised Bag of word centers
                    bagOfWordsObject.calculateSupervisedVocabulary(train_descriptors,train_labels);

                }else if(naive_bayes==true){


                    bayesClassificator->trainBayes(train_descriptors,train_labels);

                }else if(suppor_vector_machine==true){

                    suppVecMach.trainVectorMachine(train_descriptors,train_labels);

                }else if(neural_networks){

                    annClassifier->trainNeuralNetworks(train_descriptors,train_labels);

                }



                std::cout<<"Training finished."<<std::endl;

                //Save training time in a file
                training_duration = ( std::clock() - training_time ) / (double) CLOCKS_PER_SEC;

                std::cout<<"Training duration: "<<training_duration<<" sec"<<std::endl;

                char training_time_name[200];

                sprintf(training_time_name, "../results/training_time_with_%d.yml", decriptor_turn);

                cv::FileStorage training_time_file(training_time_name,cv::FileStorage::WRITE);

                training_time_file<<"Training time" << training_duration;

                training_time_file.release();

            }else{

                if(randomTrees==true){

                    //load tree if already calculated
                    rtree->loadTrainedForest(decriptor_turn);
                    std::cout<<"Tree is loaded."<<std::endl;

                }else if(bag_of_words==true){


                }else if(naive_bayes==true){

                    bayesClassificator->loadBayesData(decriptor_turn);
                    std::cout<<"Bayes data is loaded."<<std::endl;

                }else if(suppor_vector_machine==true){

                    suppVecMach.loadSVM(decriptor_turn);
                    std::cout<<"Support Vector Machine data is loaded."<<std::endl;

                }else if(neural_networks==true){

                    annClassifier->loadANN(decriptor_turn);
                    std::cout<<"Neural Network data is loaded."<<std::endl;
                }
            }

            if (test_tree){



                std::cout<<"Test started."<<std::endl;
                //saving the tree for for further use
                time_t tim;
                time(&tim);
                string treeDirectory;
                treeDirectory= homeDirectory+"results";

                string folderName = treeDirectory+"/"+ ctime(&tim);
                int status = mkdir(folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

                if(status<0){

                    std::cout<<"Error in creating Test Result folder."<<std::endl;

                }else{

                    std::cout<<"Test Result folder has been created succesfully!"<<std::endl;
                }


                //! Get test images
                Database t_im_db;
                t_im_db.setDataBase(TILDE_DB, LAPTOP,TEST);


                //                int t_loop_start =(int)std::ceil((float)t_im_db.size*((float)trainPercent/100.0f));
                //                int t_loop_finish = t_loop_start+6;
                //        int t_loop_finish = t_loop_start + (int) std::ceil((float)t_loop_start/10.0f);
                //        numberOfTestImages = (int) std::ceil((float)t_loop_start/100.0f);

                int t_loop_start = 0;
                int t_loop_finish = t_im_db.image_list.size();


                //                int n_test_im = t_loop_finish-t_loop_start;
                //                std::cout<<"Number of test images: "<<t_loop_finish-t_loop_start<<std::endl;

                cv::Mat all_res_percent, all_measured_times;


                //MEASURING DESCRIPTOR EXTRACTION + BOUNDING BOX FITTING
                std::clock_t all_start, descriptor_ext_start;
                double all_duration, descriptor_ext_duration;
                all_start = std::clock();



                //TESTING AND BOUNDINGBOX FITTING
                for(int t=t_loop_start;t<t_loop_finish;++t)
                {
                    cv::Mat measured_times(1,2,CV_64FC1);

                    //                        keypoints1.extractDescriptorsAndLables("TEST","PATCH_DIST",M_PROJECT_D,test_descriptors,test_labels,test_imgs,25,t,t+1);


                    std::cout<<"The results of test at Image "<<t<<":"<<std::endl;

                    descriptor_ext_start = std::clock();

                    keypoints1.extraction(TEST,PATCH_DIST_KP,PROJECT_2_CLASS,decriptor_turn,test_descriptors,test_labels,t,patch_radius);

                    descriptor_ext_duration = ( std::clock() - descriptor_ext_start ) / (double) CLOCKS_PER_SEC;
                    measured_times.at<double>(0,0)=descriptor_ext_duration;


                    std::cout<<"Keypoint size: "<<keypoints1.keypoints.size()<<std::endl;

                    double min=0, max=0;
                    cv::minMaxLoc(test_labels, &min, &max);
                    int class_number=(int)max+1;

                    //Initialize image matrices
                    cv::Mat results(test_descriptors.rows,1,CV_32FC1);
                    cv::Mat entropy(test_descriptors.rows,1,CV_32FC1);
                    cv::Mat res_percent(1,class_number+1,CV_32FC1);



                    if(randomTrees==true){
                        rtree->test( test_descriptors,test_labels, results, entropy, res_percent, decriptor_turn);
                        std::cout<<"Results: "<<res_percent<<std::endl;

                    }else if(bag_of_words==true){

                        bagOfWordsObject.lookUpWord(SUPERVIZED,test_descriptors,test_labels, results, entropy, res_percent);

                        std::cout<<"Results: "<<res_percent<<std::endl;

                    }else if(naive_bayes==true){

                        bayesClassificator->testBayes(test_descriptors,test_labels, results, entropy, res_percent);

                        std::cout<<"Results: "<<res_percent<<std::endl;
                    }else if(suppor_vector_machine==true){

                        suppVecMach.testVectorMachine(test_descriptors,test_labels, results, entropy, res_percent);
                        std::cout<<"Results: "<<res_percent<<std::endl;

                    }else if(neural_networks==true){

                        annClassifier->testNeuralNetworks(test_descriptors,test_labels, results, entropy, res_percent);
                        std::cout<<"Results: "<<res_percent<<std::endl;
                    }

                    all_res_percent.push_back(res_percent);

                    std::cout<<"No "<<t+1<<" Test finished."<<std::endl;
                    fflush(stdout);

                    Images im;
                    im.getImagesFromDatabase(t_im_db,t);
                    //            im = test_imgs.at(t);

                    //            cv::Mat final_im = test_imgs.at(t).col_im;
                    cv::Mat col_image = im.col_im;
                    cv::Mat extracted_facade= cv::Mat::zeros(col_image.rows,col_image.cols,col_image.type());
                    cv::Mat ent_col_image = im.col_im;
                    cv::cvtColor(ent_col_image,ent_col_image,CV_RGB2GRAY);
                    cv::cvtColor(ent_col_image,ent_col_image,CV_GRAY2BGR);
                    //std::cout<<"Category size in tree: "<<rtree-><<std::endl;

                    //Calculates entropy at each keypoint from votes
                    double max_entropy;
                    cv::minMaxIdx(entropy,0,&max_entropy);


                    for(int ent=0; ent<entropy.rows;++ent){
                        entropy.at<float>(ent,0)=entropy.at<float>(ent,0)/(float)max_entropy;
                    }


                    if(class_number==2){


                        //Colouring the keypoint to white if it is Window or Door
                        for (uint kp=0; kp< keypoints1.keypoints.size(); ++kp){
                            //at each keypoint colour



                            float res = results.at<float>(kp,0);
                            //and create a red circle if entropy high, white if it is low
                            float ent_col =entropy.at<float>(kp,0)*255;

                            cv::circle(ent_col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(255-ent_col,255-ent_col,255),2);

                            if(res==0){
                                extracted_facade.at<float>(keypoints1.keypoints[kp].pt.y, keypoints1.keypoints[kp].pt.x)=0;
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,255, 0) );
                            }else if(res==1){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,0, 255) );

                                cv::circle( extracted_facade, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 5, cv::Scalar(255,255,255),-1 );

                            }
                        }
                    }else if(class_number==4){


                        //Colouring the keypoint to white if it is Window or Door
                        for (uint kp=0; kp< keypoints1.keypoints.size(); ++kp){
                            //at each keypoint colour



                            float res = results.at<float>(kp,0);
                            //and create a red circle if entropy high, white if it is low
                            float ent_col =entropy.at<float>(kp,0)*255;

                            cv::circle(ent_col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(255-ent_col,255-ent_col,255),2);

                            if(res==0){
                                extracted_facade.at<float>(keypoints1.keypoints[kp].pt.y, keypoints1.keypoints[kp].pt.x)=0;
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,255, 0) );
                            }else if(res==1){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,0, 255) );
                                if(NUMBER_OF_CLASSES==4){
                                    cv::circle( extracted_facade, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 5, cv::Scalar(255,255,255),-1 );
                                }
                            }else if (res==2){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(255,0, 0) );
                                cv::circle( extracted_facade, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 1, cv::Scalar(0,0,0) );

                            }else if (res==3){
                                if(NUMBER_OF_CLASSES==4){
                                    cv::circle( extracted_facade, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 5, cv::Scalar(0,0,0),-1);
                                }
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,255,255) );
                            }
                        }
                    } else if(class_number==9){
                        //Colouring the keypoint to white if it is Window or Door
                        for (uint kp=0; kp< keypoints1.keypoints.size(); ++kp){
                            //at each keypoint colour

                            float res = results.at<float>(kp,0);
                            //and create a red circle if entropy high, white if it is low
                            float ent_col =entropy.at<float>(kp,0)*255;
                            //std::cout<<"Results at: "<<kp<<"  "<<res<<std::endl;
                            cv::circle(ent_col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(255-ent_col,255-ent_col,255),2);

                            if(res==0){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,0,0) );
                            }else if(res==1){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,0,128) );
                            }else if (res==2){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,128,128) );
                            }else if (res==3){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(128,0,0) );
                            }else if(res==4){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(128,0,128) );
                            }else if(res==5){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(128,128,128) );
                            }else if (res==6){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,64,128) );
                            }else if (res==7){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(128,128,0) );
                            }else if (res==8){
                                cv::circle( col_image, cv::Point(keypoints1.keypoints[kp].pt.x, keypoints1.keypoints[kp].pt.y), 2, cv::Scalar(0,128,0) );
                            }
                        }
                    }

                    //fitting the bounding box in this function
                    //                    BoundingBox bb;
                    //                    bb.get_bounding_box(extracted_facade,col_image);

                    all_duration = ( std::clock() - all_start ) / (double) CLOCKS_PER_SEC;
                    measured_times.at<double>(0,1)=all_duration;
                    all_measured_times.push_back(measured_times);

                    //Saving all the images into folder

                    char entr_name[200];
                    char im_name[200];
                    char extr_name[200];
                    //            char ext_im_name[200];
                    sprintf(entr_name, "%d_class_im_entropy_%d.png",class_number, t);
                    sprintf(im_name, "%d_class_result_%d.png",class_number, t);
                    sprintf(extr_name, "%d_extraction_result_%d.png",class_number, t);
                    //            sprintf(ext_im_name, "%d_class_result_%d.png",NUMBER_OF_CLASSES, t);
                    string im_fileName = folderName+"/"+string(im_name);
                    string entr_fileName = folderName+"/"+string(entr_name);
                    string extr_fileName = folderName+"/"+string(extr_name);

                    imshow( "Results", col_image );
                    imwrite ( im_fileName, col_image );

                    imshow( "Entropy", ent_col_image );
                    imwrite ( entr_fileName, ent_col_image );

                    imshow( "Extract", extracted_facade );
                    imwrite ( extr_fileName, extracted_facade );



                    std::cout<<"printf: "<< all_duration <<'\n';
                    std::cout<<"Finished Test on image: "<< t <<'\n';
                    cv::waitKey();

                    std::cout<<"No "<<t+1<<" Round completed"<<std::endl;

                }

                Mat av_percent, av_times;

                cv::reduce(all_res_percent,av_percent,0,CV_REDUCE_AVG,-1);
                cv::reduce(all_measured_times,av_times,0,CV_REDUCE_AVG,-1);

                char percent_file_name[200], av_time_file_name[200];

                sprintf(percent_file_name, "../results/true_positive_percentages_with_%d.yml", decriptor_turn);
                sprintf(av_time_file_name, "../results/average_time_with_%d.yml",decriptor_turn);

                cv::FileStorage percent_file(percent_file_name,cv::FileStorage::WRITE);
                cv::FileStorage average_time_file(av_time_file_name,cv::FileStorage::WRITE);

                percent_file<<"Average percentages "<<av_percent;
                percent_file<<"All percentages"<<all_res_percent;

                average_time_file<<"Average times " << av_times;

                percent_file.release();
                average_time_file.release();
            }
        }


    }

    else if(evaluation){


        std::vector<Images> trainig_imgs;
        //    int radius =0;

        //

        //        //            loading imagery
        //        ImageProcessing::loadMultipleImages(trainig_imgs, 0);
        //        ImageProcessing::loadMultipleImages(test_imgs, 1);


        std::cout<<"Image loading starts"<<std::endl;
        //! Get test images
        Database t_im_db;
        t_im_db.setDataBase(BRIGHTON_DB, LAPTOP,TEST);

        int end_it =5;

        if(t_im_db.database_id==TILDE_DB){

            end_it = t_im_db.size;
        }

        for(int tim_it=0;tim_it<end_it;++tim_it){
            Images one_im;
            one_im.getImagesFromDatabase(t_im_db,tim_it);
            trainig_imgs.push_back(one_im);
        }

        std::cout<<"Images are loaded"<<std::endl;

        std::cout<<"Evaluation started!!"<< std::endl;

        Dialog w;
        BarDialog wb;

        //! EVALUATION TYPES: INTENSITY_TEMP, INTENSITY_CHANGE, INTENSITY_SHIFT, BLUR,
        //!                   ROTATE, SIZE, AFFINE, TILDE_EVAL

        int evaluation_type = AFFINE;

        //! MATCHER TYPES: "BRUTE_FORCE", "FLANN" ,"RAW_EUCLID"(only for the same coordinates), "RAW_EUCLID_KNN"
        int matcher_type = BRUTE_FORCE;

        //! COMPARATION METHOD:"DESCRIPTOR_INDEX"

        int comparation_method = DESCRIPTOR_INDEX;

        SegmentationEvaluation seg_proc1,seg_proc2,seg_proc3,seg_proc4, seg_proc5;


        seg_proc1.setVariables(SIFT_D,matcher_type,evaluation_type,comparation_method);
        seg_proc1.evaluation(trainig_imgs);
        w.sift_plot_vector = seg_proc1.plot_vector;
        wb.time_plot_vector.push_back(seg_proc1.descriptor_count_time);

        seg_proc2.setVariables(SURF_D,matcher_type,evaluation_type,comparation_method);
        seg_proc2.evaluation(trainig_imgs);
        w.surf_plot_vector = seg_proc2.plot_vector;
        wb.time_plot_vector.push_back(seg_proc2.descriptor_count_time);

        seg_proc3.setVariables(ORB_D,matcher_type,evaluation_type,comparation_method);
        seg_proc3.evaluation(trainig_imgs);
        w.orb_plot_vector = seg_proc3.plot_vector;
        wb.time_plot_vector.push_back(seg_proc3.descriptor_count_time);

        seg_proc4.setVariables(BRIEF_D,matcher_type,evaluation_type,comparation_method);
        seg_proc4.evaluation(trainig_imgs);
        w.brief_plot_vector = seg_proc4.plot_vector;
        wb.time_plot_vector.push_back(seg_proc4.descriptor_count_time);

        seg_proc5.setVariables(D_PROJECT_D,matcher_type,evaluation_type,comparation_method);
        seg_proc5.evaluation(trainig_imgs);
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
        w.project_plot_vector = seg_proc5.plot_vector;
        wb.time_plot_vector.push_back(seg_proc5.descriptor_count_time);

        w.plot(evaluation_type);
        wb.plotBar();

        //!=================================================================


        w.show();
        wb.show();

        //        char good_pair_im_name[200];
        //        char altered_im_name[200];
        //        sprintf(good_pair_im_name, "result_imagepair.png");
        //        sprintf(altered_im_name, "altered_image.png");
        //        string good_pair_im_fileName = "../plot/"+string(good_pair_im_name);
        //        string altered_im_fileName = "../plot/"+string(altered_im_name);

        //        imwrite ( good_pair_im_fileName, seg_proc5.img_matches );
        //        imwrite ( altered_im_fileName, seg_proc5.altered_im);

        //        imshow( "Good Matches", seg_proc5.img_matches );
        //        std::cout<<"Got until here!"<<std::endl;

        waitKey();
        //        return a.exec();

    }


    else if(genetic){

        //        Genetic optimization;



        Genetic::optimize();
    }

    else if(tilde){

        std::cout<<"TILDE keypoint detector comparision is running"<<std::endl;


        //        TestTilde tt;

        //        tt.evaluate();




        //        return a.exec();

    }

    else if(evaluation2){

        std::vector<int>_genome, _genome1, _genome2, _genome3, _genome4;

        /*_genome2 ={1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,0,0,
                   0,1,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,1,1,0,
                   0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,0,
                   0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,
                   0,0,1,1,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,
                   1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1,
                   1,0,0,0,1,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,
                   0,1,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,
                   1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
                   0,1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,0,0,1,0,
                   1,0,1,1,0,1,0,1,0,0,1,1,1,0,1,1,1,1,0,1,
                   1,1,0,0,0,1,1,0,0,0,1};

        _genome1 = {1,0,1, // Color space
                    0,0,0, // 2 - Gaussian kernel size
                    1,0,1, // 3 - Patch Radius size
                    0,0,1, // 4 - RTree depth
                    0,1,0, // 5 - Number of Trees
                    1,1,1, // 6 - Randomly selected vars
                    0,0,0, // 7 - Blur sigma
                    1,     // 8 - Canny Kernel
                    0,0,0, // 9 - Canny low threshold
                    0,     // 10 - Sobel or Sharr
                    1,0,1, // 11 - tolerance
                    //======================
                    1,1, // 1 - Position
                    1,1,1,0,1,1, // 2 - Mean & std deviation
                    1,1,0,0,0,0,0,1,1, // 3 - 2nd Cent Moments
                    0,0,1,0,0,1,0,1,0,1,1,0, // 4 - 3rd Cent Moments
                    0,0,0,1,1,1,0,1,0,1,1,1,0,1,0, // 5 - 4th Cent Moments
                    1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,1,0,1,
                    0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,
                    1,1,0,0,0,1,
                    0,1,1,1,1,0,
                    0,0,1,1,
                    1,0,0,0,0,0,1,1,0,0,
                    1,1,1,0,0,1,0,1,1,0,
                    0,0,0,0,0,1,0,0,0,0,
                    1,0,
                    0,1,1,0,1,1,0,0,1,
                    1,1,0,0,0,1,1,1,1,1,
                    0,0,0,1,1,0,1,1,1,0,
                    0,0,1,0,1,0,1,1,0,1,
                    0,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,0,1};

        _genome = {0,1,1, // Color space
                   0,1,1, // Patch Radius
                   1,1,1, // Kernel size
                   1,1,0, // Sigma
                   1,0,1, // Tree depth
                   0,1,0, // Forest size
                   0,1,0, // Best split
                   1, // Canny kernel size
                   1,0,1, // Canny threshold
                   0, // Gradient calculator
                   1,1, // Position
                   1,0,1,1,1,0, // Mean and STD of channels
                   0,1,1,0,0,0,1,1,0, // 2nd cent mom
                   0,0,0,1,0,1,0,0,1,1,0,1, // 3rd cent mom
                   1,0,1,0,1,0,0,0,0,1,0,1,0,1,1, //4th cent mom
                   1,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1,0,1, // 5th cent mom
                   0,1,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0, // Hu moments
                   1,0,1,1,0,1, // Gevers L feature
                   1,0,0,1,1,1, // Gevers C feature
                   0,0,0,1, // Goesbroek C an H features
                   0,1,0,1,0,0,0,1,1,1, // Affine
                   1,0,1,0,0,1,0,1,0,0, // Affine
                   1,0,0,1,1,0,0,1,1,0, // Affine
                   1,1, // Distance Transform
                   1,0,0,1,1,0,1,1,0, // Eigen Values
                   0,1,0,1,0,0,1,0,1,0,0,1,1,1,1,                   0,0,1,0,1,0,1,1,0,1,
                   0,0,0,0,1,1,0,0,0,0,
                   1,0,1,0,0,1,0,0,0,0};  // Gradient averages


                   _genome3 = {1,0,1, // Color space --> HLS
                              0,1,0, // Kernel size--> 5
                              1,1,1, // Patch radius--> 10
                              1,1,1, // Tree depth--> 40
                              0,0,0, // Forest size--> 40
                              0,1,0, // Split number--> 12
                              1,0,1, // Sigma--> 12
                              1,  // Canny kernel--> 5
                              0,0,1, //Canny threshold--> 25
                              0, // Gradient detector--> Sobel
                              1,0,1, // Tolerance--> 60
                              1,1,
                              1,1,1,1,0,1,
                              0,1,0,0,1,1,0,0,0,
                              0,0,1,1,1,1,0,0,1,1,1,0,
                              1,0,1,0,0,0,0,0,0,0,1,1,1,1,0,
                              0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,
                              0,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,0,0,0,
                              1,0,0,1,1,1,
                              1,1,1,0,1,0,
                              0,0,0,0,
                              0,0,1,0,0,1,0,0,0,1,
                              0,1,1,1,1,0,0,0,1,0,
                              0,0,1,0,1,0,1,0,0,0,
                              1,1,
                              1,0,0,0,1,1,0,0,0,
                              1,1,1,0,1,1,0,0,1,1,
                              0,0,1,1,1,0,0,1,0,1,
                              0,0,1,0,1,0,0,1,1,0,
                              1,1,0,0,1,0,0,1,1,1,
                              0,0,0,0,0};

                   _genome4= {1,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,1,1,
                              0,1,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,
                              0,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,
                              1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,
                              1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,0};
                              */

        //PostGeneticEvaluation pge(BRIGHTON_DB,5,LAPTOP);
        //pge.genome(_genome4);
        //pge.evaluate();

    }

    else if(evaluation3){

        std::vector<std::string> source;
        source.push_back("../csv/optim1_genes.csv");
        source.push_back("../csv/optim2_genes.csv");
        source.push_back("../csv/optim3_genes.csv");

        for(int i=1; i<2; ++i){
            ParsingCSV reader(source[i], " ");

            // Get the data from CSV File
            std::vector<std::vector<int> > dataList = reader.getData();

            std::cout<<"Gene "<< dataList.size() <<std::endl;

            std::vector<int> current_vector = dataList[0];
            std::vector<int> new_vector;
            PostGeneticBundleEvaluation pge(BRIGHTON_DB,5,LAPTOP);
            std::vector<int> changing_point;

            //char result_file_file_name[200];
            //sprintf(result_file_file_name, "../results/test_results_at_8.txt");


            for(int j=0;j<dataList.size();++j){

                new_vector = dataList[j];
                std::cout<<"Iteration: "<<j<<std::endl;
                if(current_vector!=new_vector || j==0){
                    if(j==0){
                        std::cout<<"j==0"<<std::endl;
                        ofstream myfile;
                        myfile.open ("../results/test_results_at_8.txt",std::ios_base::app);
                        myfile<<"=================================== RESULTS AT ITERATION: "<<j<<"============================================="<<"\n";
                        myfile.close();
                    }else{
                        current_vector = new_vector;
                        changing_point.push_back(j);
                        ofstream myfile;
                        myfile.open ("../results/test_results_at_8.txt",std::ios_base::app);
                        myfile<<"\n=================================== RESULTS AT ITERATION: "<<j<<"============================================="<<"\n";
                        myfile.close();

                        std::cout<<"Iteration when is different"<<std::endl;
                    }
                }else{
                    std::cout<<"Yes they are the same!!!! in round "<<j<<std::endl;
                    continue;
                }

                pge.genome(new_vector);
                pge.evaluate();

            }

            std::cout<<"Changing points: (";

            for(int i = 0; i < changing_point.size(); i++) {
                std::cout << changing_point.at(i) << ',';
            }
            std::cout<<endl;
        }
        //reader.printVector(dataList[0]);

    }
}
