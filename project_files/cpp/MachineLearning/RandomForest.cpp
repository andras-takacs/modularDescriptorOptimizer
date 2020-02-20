#include "MachineLearning/RandomForest.h"


using namespace cv;
using namespace std;

RandomForest::RandomForest()
{
    //rtree = RTrees::create();
}

RandomForest::~RandomForest()
{
    if (rtree != NULL)
        rtree.release();
}

/*
float CvRTreesMultiClass::predict_multi_class( const CvMat* sample,
                                               cv::AutoBuffer<int>& out_votes,
                                               const CvMat* missing) const
{
    int result = 0;
    int k;

    if( nclasses > 0 ) //classification
    {
        int max_nvotes = 0;
        int* votes = out_votes;
        memset( votes, 0, sizeof(*votes)*nclasses );
        for( k = 0; k < ntrees; k++ )
        {
            //Do the prediction at the keyfram
            CvDTreeNode* predicted_node = trees[k]->predict( sample, missing );
            int nvotes;
            int class_idx = predicted_node->class_idx;
            CV_Assert( 0 <= class_idx && class_idx < nclasses );

            //Get the votes from each tree
            nvotes = ++votes[class_idx];
            if ( nvotes > max_nvotes )
            {
                max_nvotes = nvotes;
                result = predicted_node->value;
            }
        }
    }
    else // regression
    {
        //throw std::runtime_error(__FUNCTION__ "can only be used classification");
    }

    return result;
}

float CvRTreesMultiClass::predict_multi_class( const Mat& _sample,
                                               cv::AutoBuffer<int>& out_votes,
                                               const Mat& _missing) const

{
    CvMat sample = _sample;
    CvMat mmask = _missing;
    return predict_multi_class(&sample, out_votes, mmask.data.ptr ? &mmask : 0);
}*/

void RandomForest::loadTrainedForest(int i){
    //LOAD TREE FROM FILE
    char name[200];
    string treeDirectory;
    string fileNameTemplate;


    fileNameTemplate = "trained_tree_%d";
    treeDirectory=homeDirectory+"mldata/trees";


    sprintf(name, fileNameTemplate.c_str(), i);
    string fileName = treeDirectory+"/"+string(name);

    char cfilename[200];
    strcpy(cfilename,fileName.c_str());

    const char* filename = cfilename;
    std::cout<<filename<<std::endl;
    char tree_name[200];
    sprintf(tree_name,"trained_tree_%d",i);

    if ( rtree ) delete rtree;
    //creating an empty tree if it is not empty and loads the tree there
    //rtree = new CvRTreesMultiClass();
    Ptr<RTrees> rtree = RTrees::create();
    rtree->load(filename);

    std::cout<<"Finished loading the trees"<<std::endl;
}

void RandomForest::test(Mat const &testing_data,
                        Mat const&testing_classifications,
                        cv::Mat results,
                        cv::Mat entropy,
                        cv::Mat res_percents,
                        const int &descriptor_type)
{
    Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;


    double min=0, max=0;
    cv::minMaxLoc(testing_classifications, &min, &max);
    int class_number=(int)max+1;

    std::cout<<"Class num: "<<class_number<<std::endl;

    int false_positives[class_number];
    Mat conf_matrix = Mat::zeros(class_number,class_number,CV_32FC1);
    int true_positives[class_number];
    int total_class_result[class_number];

    for(int nc=0; nc<class_number; ++nc){
        false_positives[nc]=0;
        true_positives[nc]=0;
        total_class_result[nc]=0;
    };

    std::cout<<"Descriptor numbers "<<testing_data.rows<<std::endl;

    //printf( "\nUsing testing database: %s\n\n", argv[2]);


    //
    /* Flags for the get votes
     *  PREDICT_AUTO =0, // Returns the number of votes per class
     *  PREDICT_SUM =(1<<8), // Returns the class prediction at each tree
     *  PREDICT_MAX_VOTE =(2<<8), //
     *  PREDICT_MASK =(3<<8)
     */

    TermCriteria tcrit = rtree->getTermCriteria();
    double norm = tcrit.maxCount;
    //std::cout<<"Macx count: "<<tcrit.maxCount;

    Mat all_votes;
    rtree->getVotes(testing_data, all_votes, cv::ml::DTrees::Flags::PREDICT_AUTO);
    //std::cout<<"PREDICT AUTO: "<<all_votes<<std::endl;
    rtree->predict( testing_data, results );
    //std::cout<<"RESULTS: "<<results<<std::endl;




    int sizes[] = {4,4,4};
    cv::Mat confusion_matrices = cv::Mat::zeros(3, sizes, CV_32SC3);

    for (int tsample = 0; tsample < testing_data.rows; tsample++)
    {

        //Check the strength of the prediction
        float kp_entrophy = 0;
        for (int k =0; k < class_number; ++k)

        {
            //std::cout<<"Votes:"<<out_votes[k]<<", ";
            if(all_votes.at<int>(tsample+1,k)==0) continue;

            double v = (double) all_votes.at<int>(tsample+1,k)/norm;
            //std::cout<<"Normalized vote: "<<v<<std::endl;
            kp_entrophy =  kp_entrophy+(log2(v)*v);
            //std::cout<< "Vote: "<< all_votes.at<int>(tsample+1,k) <<std::endl;
            //std::cout << "Log Normalised vote at "<<tsample<<" : "<< std::log2(v)*v<<std::endl;

        }
        kp_entrophy = kp_entrophy*(-1);
        entropy.at<float>(tsample,0)=(float)kp_entrophy;
        //std::cout << "Enthropy at "<<tsample<<" : "<< entropy.at<float>(tsample,0) <<std::endl;

        int res = (int) results.at<float>(tsample,0);
        int clasif = testing_classifications.at<float>(tsample, 0);
        // if they differ more than floating point error => wrong class
        conf_matrix.at<float>(res,clasif)++;

        for (int i=0;i<4;++i){
            if (i==res && res==clasif) confusion_matrices.at<int>(0, 0, i)++;
            if (i!=res && res==clasif) confusion_matrices.at<int>(1, 1, i)++;
            if (i==res && i!=clasif) confusion_matrices.at<int>(0, 1, i)++;
            if (i!=res && i==clasif) confusion_matrices.at<int>(1, 0, i)++;
        }

        // (N.B. openCV uses a floating point decision tree implementation!)
        //std::cout<<"RES - PRIOR: "<<fabs(result - testing_classifications.at<float>(tsample, 0))<<std::endl;

        if (fabs(results.at<float>(tsample,0) - testing_classifications.at<float>(tsample, 0))
                >= FLT_EPSILON)
        {


            wrong_class++;

            //cout << "WRONG CLASS"<< (int) results.at<float>(tsample,0) << endl;
            false_positives[(int) results.at<float>(tsample,0)]++;

        }
        else
        {

            // otherwise correct

            correct_class++;

            true_positives[(int) results.at<float>(tsample,0)]++;
        }
    }

    for (int i = 0; i < class_number; ++i)
    {
        total_class_result[i] = false_positives[i]+true_positives[i];
    }


    Mat measurements = cv::Mat::zeros(6,4,CV_32FC2);

    for(int i=0;i<4; ++i){
        int TP = confusion_matrices.at<int>(0, 0, i);
        int TN = confusion_matrices.at<int>(1, 1, i);
        int FP = confusion_matrices.at<int>(0, 1, i);
        int FN = confusion_matrices.at<int>(1, 0, i);
        int totalPop = TP+TN+FP+FN;


        float accuracy = ((float)TP+(float)TN)/(float)totalPop;
        float PPV = (float)TP/((float)TP+(float)FP);
        float TPR = (float)TP/((float)TP+(float)FN);
        float TNR = (float)TN/((float)TN+(float)FP);
        float NPV = (float)TN/((float)TN+(float)FN);
        float f1_measure = 2*(PPV*TPR)/(PPV+TPR);

        measurements.at<float>(0,i) =(float) accuracy;
        measurements.at<float>(1,i) = PPV;
        measurements.at<float>(2,i) = TPR;
        measurements.at<float>(3,i) = TNR;
        measurements.at<float>(4,i) = NPV;
        measurements.at<float>(5,i) = f1_measure;

//        std::cout<<"Accuracy: "<<measurements.at<float>(0,i)<<std::endl;
    }

    //=================Write results into text file===========================


    char result_file_file_name[200];

    sprintf(result_file_file_name, "../results/test_results_at_%d.txt", descriptor_type);


    ofstream myfile;
    time_t tim;
    time(&tim);
    myfile.open (result_file_file_name,std::ios_base::app);
    //myfile<<"\nReults on: "<<ctime(&tim)<<"\n";
/*
    if(class_number==3){
        myfile <<"Classes: (2)Wall (1)Window,Door,Roof (0)Everything else\n";
    }else if(class_number==4){
        myfile <<"Classes: (3)Wall (2)Roof (1)Window,Door (0)Everything else\n";
    }

    myfile<< "Descriptor Feature size: "<<testing_data.cols<<" and "<<NUMBER_OF_CLASSES<<" Classes, with "<<number_of_images<<" images\n"
          << "Total correct (TC) classification: "<<correct_class<<" ("<<(double) correct_class*100/testing_data.rows <<"%)\n"
          <<"Total wrong (TW) classifications: "<<wrong_class<<" ("<<(double) wrong_class*100/testing_data.rows<<"%)\n"
         <<"The correctly classified points in detail:\n";

    for (int i = 0; i < class_number; ++i)
    {
        myfile <<"From class ("<<i<<") "<<"the number of correctly classified points: "<< true_positives[i] <<" is ("<<(double)true_positives[i]*100/total_class_result[i]
                 <<"%) of Class("<<i<<") points:\n";
    }
    myfile <<"The wrongly classified points in detail:\n";

    for (int i = 0; i < class_number; ++i)
    {
        myfile <<"From class ("<<i<<") "<<"the number of wrongly classified points: "<< false_positives[i] <<" is ("<<(double)false_positives[i]*100/total_class_result[i]
                 <<"%) of Class("<<i<<") points:\n"
                <<"\t from which:\n";
        for(int j=0; j<class_number; ++j){
            if(i==j) continue;

            myfile<<(int)conf_matrix.at<float>(i,j)<<" classified as Class ("<<j<<")\n";
        }
    }
    myfile<<"CONFUSION MATRIX\n";

    if(class_number==3) myfile<<"      ELSE  DWR  WALL\n";
    if(class_number==4) myfile<<"      ELSE  D&W  ROOF  WALL\n";
*/
    for(int i=0;i<class_number;++i){
        /*
        if(class_number==3){
            if(i==0) myfile<<"ELSE |";

            if(i==1) myfile<<"DWR  |";

            if(i==2) myfile<<"WALL |";
        }
        if(class_number==4){
            if(i==0) myfile<<"ELSE |";

            if(i==1) myfile<<"D&W  |";

            if(i==2) myfile<<"ROOF |";

            if(i==3) myfile<<"WALL |";
        }
*/
        for(int j=0;j<class_number;++j){
            if (i==j){
                myfile<<true_positives[i]<<", ";
            }else{
                myfile<<conf_matrix.at<float>(i,j)<<", ";
            }
        }
        myfile<<"\n";
    }

    for(int i=0;i<class_number;++i){


        myfile<<"Measurements at CLASS ("<<i<<")\n";
        myfile<<"Accuracy: "<<measurements.at<float>(0, i)<<"\n";
        myfile<<"Precision: "<<measurements.at<float>(1, i)<<"\n";
        myfile<<"Recall: "<<measurements.at<float>(2, i)<<"\n";
        myfile<<"Specificity: "<<measurements.at<float>(3, i)<<"\n";
        myfile<<"Negative Predictive Value: "<<measurements.at<float>(4, i)<<"\n";
        myfile<<"F-Measure: "<<measurements.at<float>(5, i)<<"\n";

    }

    myfile.close();


    /*
    //=================APPLICATION OUTPUT===================
    if(NUMBER_OF_CLASSES==3){
        printf( "\tClasses: (2)Wall (1)Window,Door,Roof (0)Everything else with %d images\n",number_of_images);
    }else if(NUMBER_OF_CLASSES==4){
        printf( "\tClasses: (3)Wall (2),Roof (1)Windo w,Door (0)Everything else with %d images\n",number_of_images);
    }

    printf( "\tTotal correct (TC) classification: %d (%g%%)\n"
            "\tTotal wrong (TW) classifications: %d (%g%%)\n",
            correct_class, (double) correct_class*100/testing_data.rows,
            wrong_class, (double) wrong_class*100/testing_data.rows);
*/
    res_percents.at<float>(0,0)=(float) correct_class*100/testing_data.rows;

    //    printf("\tThe correctly classified points in detail:\n");

    for (int i = 0; i < class_number; ++i)
    {

        //        printf( "\t From class (%d) the number of correctly classified points: %d is (%g%%) of Class(%d) points:\n", i,
        //                true_positives[i],
        //                (double) true_positives[i]*100/total_class_result[i],i);
        if(true_positives[i] ==0 || total_class_result[i]==0){
        res_percents.at<float>(0,i+1)=0.0;
        }else{
        res_percents.at<float>(0,i+1)=(float) true_positives[i]*100/total_class_result[i];
        }
        //        std::cout<<"True positive at class "<<i<<": "<<(double) true_positives[i]<<" Class results: "<<total_class_result[i]<<std::endl;
    }

    //    printf("\tThe wrongly classified points in detail:\n");
    /*
    for (uint i = 0; i < NUMBER_OF_CLASSES; ++i)
    {
        printf( "\t From class (%d) the number of wrongly classified points: %d is (%g%%) of Class(%d) points:\n"
                "\t from which:\n", i,
                false_positives[i],
                (double) false_positives[i]*100/total_class_result[i],i);
        for(uint j=0; j<NUMBER_OF_CLASSES; ++j)
        {
            if(i==j) continue;

            printf("\t %d classified as Class (%d)\n",(int)conf_matrix.at<float>(i,j),j);
        }
    }
    std::cout<<"CONFUSION MATRIX\n";

    if(NUMBER_OF_CLASSES==3) std::cout<<"      ELSE  DWR  WALL\n";
    if(NUMBER_OF_CLASSES==4) std::cout<<"      ELSE  D&W  ROOF  WALL\n";

    for(uint i=0;i<NUMBER_OF_CLASSES;++i){
        if(NUMBER_OF_CLASSES==3){
            if(i==0) std::cout<<"ELSE |";

            if(i==1) std::cout<<"DWR  |";

            if(i==2) std::cout<<"WALL |";
        }
        if(NUMBER_OF_CLASSES==4){
            if(i==0) std::cout<<"ELSE |";

            if(i==1) std::cout<<"D&W  |";

            if(i==2) std::cout<<"ROOF |";

            if(i==3) std::cout<<"WALL |";
        }
        for(uint j=0;j<NUMBER_OF_CLASSES;++j){
            if (i==j){
                std::cout<<true_positives[i]<<", ";
            }else{
                std::cout<<conf_matrix.at<float>(i,j)<<", ";
            }
        }
        std::cout<<"\n";
    }*/
}

void RandomForest::train(cv::Mat const&training_data,
                         cv::Mat const&training_classifications,
                         const int &descriptor_type,
                         const int tree_depth,
                         const int forest_size,
                         const int number_for_split)
{
    std::cout << "training rtree" << std::endl;
    float _priors[] = {1.0,1.0,1.0,1.0};  // weights of each classification for classes
    Mat priors( 1, 4, CV_32F, _priors );
    // (all equal as equal samples of each digit)

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis
    const int ATTRIBUTES_PER_SAMPLE = training_data.cols;
    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type.setTo(Scalar(VAR_NUMERICAL) ); // all inputs are numerical


    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = VAR_CATEGORICAL;

//!    [param] Maximum depth of the tree (-max): The depth of the tree. A low value will likely underfit and conversely
//!            a high value will likely overfit. The optimal value can be obtained using cross validation or other suitable methods.
//!    [param] Minimum number of samples in each node (-min): If the number of samples in a node is smaller than this parameter,
//!            then the node will not be split. A reasonable value is a small percentage of the total data e.g. 1 percent.
//!    [param] Termination Criteria for regression tree (-ra): If all absolute differences between an estimated value in a node
//!            and the values of the train samples in this node are smaller than this regression accuracy parameter, then the node will not be split.
//!    [param] Cluster possible values of a categorical variable into K <= cat clusters to find a suboptimal split (-cat):
//!            Cluster possible values of a categorical variable into K <= cat clusters to find a suboptimal split.
//!    [param] Size of the randomly selected subset of features at each tree node (-var): The size of the subset of features,
//!            randomly selected at each tree node, that are used to find the best split(s). If you set it to 0, then the size will be set to
//!            the square root of the total number of features.
//!    [param] Maximum number of trees in the forest (-nbtrees): The maximum number of trees in the forest. Typically,
//!            the more trees you have, the better the accuracy. However, the improvement in accuracy generally diminishes and reaches an asymptote
//!            for a certain number of trees. Also to keep in mind, increasing the number of trees increases the prediction time linearly.
//!    [param] Sufficient accuracy (OOB error) (-acc): Sufficient accuracy (OOB error).

    /*CvRTParams params = CvRTParams(tree_depth, // max depth default: 20
                                   5, // min sample count
                                   0, // regression accuracy: N/A here
                                   false, // compute surrogate split, no missing data
                                   15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                   priors, // the array of priors
                                   false,  // calculate variable importance
                                   number_for_split,       // number of variables randomly selected at node and used to find the best split(s).--> before 4
                                   forest_size,	 // max number of trees in the forest--->Briemann has 100
                                   0.01f,				// forrest accuracy
                                   CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                   );*/

    // train random forest classifier (using training data)

    //printf( "\nUsing training database: %s\n\n", argv[1]);
    if ( rtree ) delete rtree;
    int nsamples_all = training_data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);
    Ptr<TrainData> tdata = prepare_train_data(training_data, training_classifications, ntrain_samples);
    rtree = RTrees::create();
    //rtree->setMaxDepth(10);
    rtree->setMaxDepth(tree_depth);
    rtree->setMinSampleCount(10);
    rtree->setRegressionAccuracy(0);
    rtree->setUseSurrogates(false);
    rtree->setMaxCategories(15);
    rtree->setPriors(priors);
    rtree->setCalculateVarImportance(true);
    rtree->setActiveVarCount(number_for_split);
    rtree->setActiveVarCount(5);
    rtree->setTermCriteria(TermCriteria(TermCriteria::COUNT, forest_size, 1e-6 ));
    //rtree->setTermCriteria(TermCriteria(TermCriteria::COUNT, 10, 1e-6 ));
    rtree->train(tdata);



    /*rtree->train(training_data,  training_classifications,
                 Mat(), Mat(), var_type, Mat(), params);*/

    //============================SAVING THE TREE=============================
    char tree_name[200];
    sprintf(tree_name,"../mldata/trees/trained_tree_%d",descriptor_type);
    rtree->save(tree_name);

    TermCriteria tcrit = rtree->getTermCriteria();
    double num_of_trees = tcrit.maxCount;
    std::cout<<"Forest has been saved with "<<num_of_trees<<" trees"<<std::endl;
}

Ptr<TrainData> RandomForest::prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    std::cout<<"Matrix dimesions: "<<data.dims<<std::endl;

    return TrainData::create(data, ROW_SAMPLE, responses, noArray(), sample_idx, noArray(), var_type);
}
