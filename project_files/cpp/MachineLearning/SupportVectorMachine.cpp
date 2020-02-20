#include "MachineLearning/SupportVectorMachine.h"

SupportVectorMachine::SupportVectorMachine()
{

}

SupportVectorMachine::~SupportVectorMachine()
{

}

void SupportVectorMachine::trainVectorMachine(Mat const& training_data, Mat const& training_labels){


    //!==================================DEFAULT PARAM SETUP==========================
    //!CvSVMParams::CvSVMParams():svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0),
    //!                           gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0){
    //!term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );}
    //!
    //!============================SVM TYPES==========================================
    //!CvSVM::C_SVC ==== C-Support Vector Classification. n-class classification (n \geq 2),
    //!     allows imperfect separation of classes with penalty multiplier C for outliers.
    //!
    //!CvSVM::NU_SVC ==== \nu-Support Vector Classification. n-class classification with possible imperfect separation.
    //!     Parameter \nu (in the range 0..1, the larger the value, the smoother the decision boundary) is used instead of C.
    //!
    //!CvSVM::ONE_CLASS ==== Distribution Estimation (One-class SVM). All the training data are from the same class,
    //!     SVM builds a boundary that separates the class from the rest of the feature space.
    //!
    //!CvSVM::EPS_SVR ==== \epsilon-Support Vector Regression. The distance between feature vectors from the training set
    //!     and the fitting hyper-plane must be less than p. For outliers the penalty multiplier C is used.
    //!
    //!CvSVM::NU_SVR ==== \nu-Support Vector Regression. \nu is used instead of p.
    //!
    //!=================================KERNEL TYPE====================================
    //!CvSVM::LINEAR ==== Linear kernel. No mapping is done, linear discrimination (or regression)
    //!     is done in the original feature space. It is the fastest option. K(x_i, x_j) = x_i^T x_j.
    //!
    //!CvSVM::POLY ==== Polynomial kernel: K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0.
    //!
    //!CvSVM::RBF ==== Radial basis function (RBF), a good choice in most cases.
    //!     K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0.
    //!
    //!CvSVM::SIGMOID ==== Sigmoid kernel: K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).
    //!
    //!===================================OTHER PARAMETERS=============================
    //!degree – Parameter degree of a kernel function (POLY).
    //!
    //!gamma – Parameter \gamma of a kernel function (POLY / RBF / SIGMOID).
    //!
    //!coef0 – Parameter coef0 of a kernel function (POLY / SIGMOID).
    //!
    //!Cvalue – Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
    //!
    //!nu – Parameter \nu of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
    //!
    //!p – Parameter \epsilon of a SVM optimization problem (EPS_SVR).
    //!
    //!class_weights – Optional weights in the C_SVC problem , assigned to particular classes.
    //!     They are multiplied by C so the parameter C of class #i becomes class\_weights_i * C.
    //!     Thus these weights affect the misclassification penalty for different classes.
    //!     The larger weight, the larger penalty on misclassification of data from the corresponding class.
    //!
    //!term_crit – Termination criteria of the iterative SVM training procedure which solves a partial case of
    //!     constrained quadratic optimization problem. You can specify tolerance and/or the maximum number of iterations.

    // Set up SVM's parameters
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));


    if (training_data.data == NULL || training_labels.data == NULL){
        cerr << "SVN training matrices are emprty(NULL)!!" << endl;
    } else {
        svm->clear();
        cout << "Starting training Support Vector Machine" << endl;
        svm->train(training_data, ROW_SAMPLE, training_labels);
    }


}

void SupportVectorMachine::testVectorMachine(Mat const& testing_data, Mat const& testing_labels, Mat results, Mat entropy, Mat res_percents){

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

        // run vector machine prediction
        result = svm->predict(test_sample);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point vector machine implementation!)

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

    for (int i = 0; i < max_labels; ++i)
    {
        total_class_result[i] = false_positives[i]+true_positives[i];
    }

    res_percents.at<float>(0,0)=(float) correct_class*100/testing_data.rows;

    for (int i = 0; i < max_labels; ++i)
    {
        res_percents.at<float>(0,i+1)=(float) true_positives[i]*100/total_class_result[i];

    }
}

void SupportVectorMachine::loadSVM(int descriptor_type){

    //LOAD VECTOR MACHINE FROM FILE
    char name[200];
    string svmDirectory;
    string fileNameTemplate;


    fileNameTemplate = "trained_svm_%d";
    svmDirectory=homeDirectory+"mldata/svm";


    sprintf(name, fileNameTemplate.c_str(), descriptor_type);
    string fileName = svmDirectory+"/"+string(name);

    char cfilename[200];
    strcpy(cfilename,fileName.c_str());

    const char* filename = cfilename;
    std::cout<<filename<<std::endl;
    char svm_name[200];
    sprintf(svm_name,"trained_svm_%d",descriptor_type);

    svm->clear();
    //creating an empty Vector Machine if it is not empty and loads the tree there
    svm->load(filename);

    std::cout<<"Finished loading the SVM data"<<std::endl;
}


void SupportVectorMachine::saveSVM(int descriptor_type){

    char svm_name[200];
    sprintf(svm_name,"../mldata/svm/trained_svm_%d",descriptor_type);
    svm->save(svm_name);

}
