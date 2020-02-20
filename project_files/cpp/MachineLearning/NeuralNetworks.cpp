#include "MachineLearning/NeuralNetworks.h"

NeuralNetworks::NeuralNetworks()
{

}

NeuralNetworks::~NeuralNetworks()
{

    if( nnetwork ) delete nnetwork;

}

void NeuralNetworks::trainNeuralNetworks(const Mat &training_data, const Mat &training_labels){


    // define the structure for the neural network (MLP)
    // The neural network has 3 layers.
    // - one input node per attribute in a sample so 256 input nodes
    // - 16 hidden nodes
    // - 10 output node, one for each class.

    int training_data_cols = training_data.cols;

    double min=0, max=0;
    cv::minMaxLoc(training_labels, &min, &max);
    int class_number=(int)max+1;

    cv::Mat layers(3,1,CV_32S);
    layers.at<int>(0,0) = training_data_cols;//input layer
    layers.at<int>(1,0)=16;//hidden layer
    layers.at<int>(2,0) =class_number;//output layer



    //create the neural network.
    //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html

    if (training_data.data == NULL || training_labels.data == NULL){
        cerr << "ANN training matrices are empty(NULL)!!" << endl;
    } else {
    nnetwork->clear();
    nnetwork = ANN_MLP::create();
    }

    int nsamples_all = training_data.rows;
    int ntrain_samples = (int)(nsamples_all*0.8);
    Ptr<TrainData> tdata = prepare_train_data(training_data, training_labels, ntrain_samples);

    nnetwork->setLayerSizes(layers);
    nnetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    nnetwork->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, FLT_EPSILON));
    nnetwork->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
    nnetwork->train(tdata);

    /*
    nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);

    CvANN_MLP_TrainParams params(

                // terminate the training after either 1000
                // iterations or a very small change in the
                // network wieghts below the specified value
                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
                // use backpropogation for training
                CvANN_MLP_TrainParams::BACKPROP,
                // co-efficents for backpropogation training
                // recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
                0.1,
                0.1);
    */

    cout << "Starting training Neural Networks" << endl;
    //int iterations = nnetwork->train(training_data, training_labels, Mat(), Mat(), params);

    //printf( "Training iterations: %i\n\n", iterations);


}

void NeuralNetworks::testNeuralNetworks(Mat const& testing_data, Mat const& testing_labels, Mat results, Mat entropy, Mat res_percents){

    int num_desc = testing_data.rows;
    double min=0, max=0;
    cv::minMaxLoc(testing_labels, &min, &max);
    int class_numbers=(int)max+1;

    Mat classificationResult = Mat(1, class_numbers, CV_32FC1);
    Point max_loc = Point(0,0);

    // perform classifier testing and report results

    Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [class_numbers];
    int true_positives [class_numbers];
    int total_class_result [class_numbers];

    // zero the false positive counters in a simple loop

    for(int nc=0; nc<class_numbers; ++nc){
        false_positives[nc]=0;
        true_positives[nc]=0;
        total_class_result[nc]=0;
    };

    for (int tsample = 0; tsample < num_desc; tsample++)
    {
        // extract a row from the testing matrix
        test_sample = testing_data.row(tsample);


        // run neural network prediction
        nnetwork->predict(test_sample, classificationResult);

        // The NN gives out a vector of probabilities for each class
        // We take the class with the highest "probability"
        // for simplicity (but we really should also check separation
        // of the different "probabilities" in this vector - what if
        // two classes have very similar values ?)

        minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

        printf("Testing Sample %i -> class result (digit %d)\n", tsample, max_loc.x);

        // if the corresponding location in the testing classifications
        // is not "1" (i.e. this is the correct class) then record this

        int result = 0;
        float value=0.0f;
        float maxValue=classificationResult.at<float>(0,0);
        for(int index=1;index<class_numbers;index++)
        {   value = classificationResult.at<float>(0,index);
            std::cout<<"Value at loop "<<tsample<<": "<<value<<std::endl;
            if(value>maxValue)
            {   maxValue = value;
                result=index;

            }
        }


        results.at<float>(tsample,0)=(float)result;
        entropy.at<float>(tsample,0)=0;

        if (fabs(result - testing_labels.at<float>(tsample, 0))>= FLT_EPSILON){

            // if they differ more than floating point error => wrong class
            wrong_class++;

            false_positives[((int) result)]++;

        }else{

            // otherwise correct
            correct_class++;

            true_positives[(int) result]++;
        }
    }

    for (int i = 0; i < class_numbers; ++i)
    {
        total_class_result[i] = false_positives[i]+true_positives[i];
    }

    res_percents.at<float>(0,0)=(float) correct_class*100/testing_data.rows;

    for (int i = 0; i < class_numbers; ++i)
    {
        res_percents.at<float>(0,i+1)=(float) true_positives[i]*100/total_class_result[i];

    }
}

void NeuralNetworks::loadANN(int descriptor_type){


    //LOAD NEURAL NETWORK FROM FILE
    char name[200];
    string annDirectory;
    string fileNameTemplate;


    fileNameTemplate = "trained_ann_%d";
    annDirectory=homeDirectory+"mldata/ann";


    sprintf(name, fileNameTemplate.c_str(), descriptor_type);
    string fileName = annDirectory+"/"+string(name);

    char cfilename[200];
    strcpy(cfilename,fileName.c_str());

    const char* filename = cfilename;
    std::cout<<filename<<std::endl;
    char ann_name[200];
    sprintf(ann_name,"trained_ann_%d",descriptor_type);

    nnetwork->clear();
    //creating an empty Neural Network if it is not empty and loads the tree there
    nnetwork->load(filename);

    std::cout<<"Finished loading the ANN data"<<std::endl;

}

void NeuralNetworks::saveANN(int descriptor_type){

    char ann_name[200];
    sprintf(ann_name,"../mldata/ann/trained_ann_%d",descriptor_type);
    nnetwork->save(ann_name);

}
Ptr<TrainData> NeuralNetworks::prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
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

