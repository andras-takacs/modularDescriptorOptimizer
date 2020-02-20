#include "MachineLearning/NaiveBayes.h"

NaiveBayes::NaiveBayes()
{

}

NaiveBayes::~NaiveBayes(){

    if (bayes != nullptr)
        delete bayes;

}



void NaiveBayes::trainBayes(Mat& training_data, Mat& training_labels){


    if (bayes != nullptr) bayes->clear();

    if (training_data.data == NULL || training_labels.data == NULL){
        cerr << "Bayes training matrices are emprty(NULL)!!" << endl;
    } else {

        //bayes = NormalBayesClassifier::create();

        std::cout<<"Bayes training started"<<std::endl;
        std::cout<<"Descriptor size: "<<training_data.rows<<" Label size: "<<training_labels.rows<<std::endl;

        int nsamples_all = training_data.rows;
        int ntrain_samples = (int)(nsamples_all*0.8);
        Ptr<TrainData> tdata = prepare_train_data(training_data, training_labels, ntrain_samples);
        Ptr<NormalBayesClassifier> normalBayesClassifier = StatModel::train<NormalBayesClassifier>(tdata);
        //bayes = StatModel::train<NormalBayesClassifier>(training_data, training_labels, Mat(), Mat(), false);
    }

}

void NaiveBayes::testBayes(Mat const& testing_data, Mat const& testing_labels, Mat results, Mat entropy, Mat res_percents){

    int num_desc = testing_data.rows;
    double min=0, max=0;
    cv::minMaxLoc(testing_labels, &min, &max);
    int max_labels=(int)max+1;

    // perform classifier testing and report results

    Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [max_labels];
    int true_positives [max_labels];
    int total_class_result [max_labels];
    float result;

    // zero the false positive counters in a simple loop

    for(int nc=0; nc<max_labels; ++nc){
        false_positives[nc]=0;
        true_positives[nc]=0;
        total_class_result[nc]=0;
    };

    for (int tsample = 0; tsample < num_desc; tsample++)
    {

        // extract a row from the testing matrix
        test_sample = testing_data.row(tsample);

        // run decision tree prediction
        result = bayes->predict(test_sample);

        //        printf("Testing Sample %i -> class result (character %c)\n", tsample,
        //               CLASSES[((int) result)]);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)

        results.at<float>(tsample,0)=(float)result;
        entropy.at<float>(tsample,0)=0;

        if (fabs(result - testing_labels.at<float>(tsample, 0))
                >= FLT_EPSILON)
        {
            // if they differ more than floating point error => wrong class

            wrong_class++;

            false_positives[((int) result)]++;

        }
        else
        {

            // otherwise correct

            correct_class++;

            true_positives[(int) result]++;
        }
    }

    for (int i = 0; i < max_labels; ++i)
    {
        total_class_result[i] = false_positives[i]+true_positives[i];
    }

    /* printf( "\nResults on the testing database: %s\n"
            "\tCorrect classification: %d (%g%%)\n"
            "\tWrong classifications: %d (%g%%)\n",
            argv[2],
            correct_class, (double) correct_class*100/num_desc,
            wrong_class, (double) wrong_class*100/num_desc);

    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        printf( "\tClass (character %c) false postives 	%d (%g%%)\n", CLASSES[i],
                false_positives[i],
                (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
    }*/


    res_percents.at<float>(0,0)=(float) correct_class*100/testing_data.rows;

    for (int i = 0; i < max_labels; ++i)
    {
        res_percents.at<float>(0,i+1)=(float) true_positives[i]*100/total_class_result[i];

    }


}

void NaiveBayes::loadBayesData(int descriptor_type){
    //LOAD BAYES FROM FILE
    char name[200];
    string bayesDirectory;
    string fileNameTemplate;


    fileNameTemplate = "trained_bayes_%d";
    bayesDirectory=homeDirectory+"mldata/bayes";


    sprintf(name, fileNameTemplate.c_str(), descriptor_type);
    string fileName = bayesDirectory+"/"+string(name);

    char cfilename[200];
    strcpy(cfilename,fileName.c_str());

    const char* filename = cfilename;
    std::cout<<filename<<std::endl;
    char bayes_name[200];
    sprintf(bayes_name,"trained_bayes_%d",descriptor_type);

    bayes->clear();
    //clearing an Naive Bayes Object if it is not empty and loads the tree there
//    bayes = new CvNormalBayesClassifier();
    bayes->load(filename);

    std::cout<<"Finished loading the Bayes data"<<std::endl;
}

void NaiveBayes::saveBayesData(int descriptor_type){

    char bayes_name[200];
    sprintf(bayes_name,"../mldata/bayes/trained_bayes_%d",descriptor_type);
    bayes->save(bayes_name);
}

Ptr<TrainData> NaiveBayes::prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}