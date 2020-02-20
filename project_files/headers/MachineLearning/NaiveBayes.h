#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include "utils.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

class NaiveBayes
{
public:
    NaiveBayes();
    ~NaiveBayes();

    void trainBayes(Mat &training_data, Mat &training_labels);

    void testBayes(cv::Mat const& testing_data,
              cv::Mat const& testing_labels,
              cv::Mat results,
              cv::Mat entropy,
              cv::Mat res_percents);

    void saveBayesData(int descriptor_type);

    void loadBayesData(int descriptor_type);

    Ptr<TrainData> prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples);


    Ptr<NormalBayesClassifier> bayes;
};

#endif // NAIVEBAYES_H
