#ifndef NEURALNETWORKS_H
#define NEURALNETWORKS_H

#include "utils.h"
using namespace cv;
using namespace cv::ml;

class NeuralNetworks
{
public:
    NeuralNetworks();
    ~NeuralNetworks();

    void trainNeuralNetworks(const Mat &training_data, const Mat &training_labels);
    void testNeuralNetworks(Mat const& testing_data, Mat const& testing_labels, Mat results, Mat entropy, Mat res_percents);
    void loadANN(int descriptor_type);
    void saveANN(int descriptor_type);
    Ptr<TrainData> prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples);


    Ptr<ANN_MLP> nnetwork;
};

#endif // NEURALNETWORKS_H
