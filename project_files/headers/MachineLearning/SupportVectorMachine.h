#ifndef SUPPORTVECTORMACHINE_H
#define SUPPORTVECTORMACHINE_H

#include "utils.h"
using namespace cv;
using namespace cv::ml;


class SupportVectorMachine
{
public:
    SupportVectorMachine();
    ~SupportVectorMachine();

    void trainVectorMachine(const Mat &training_data, const Mat &training_labels);
    void testVectorMachine(Mat const& testing_data, Mat const& testing_labels, Mat results, Mat entropy, Mat res_percents);
    void loadSVM(int descriptor_type);
    void saveSVM(int descriptor_type);

    Ptr<SVM> svm = SVM::create();
};

#endif // SUPPORTVECTORMACHINE_H
