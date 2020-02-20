#ifndef BAGOFWORDS_H
#define BAGOFWORDS_H

#include "utils.h"

using namespace cv;
using namespace std;

class BagOfWords
{
public:
    BagOfWords();
    ~BagOfWords();

    Mat supervisedVocabulary, unSupervisedVocabulary, unsupervisedLabels;

    void calculateSupervisedVocabulary(Mat& _all_descriptors, Mat& _marked_labels);
    void calculateUnSupervisedVocabulary(Mat& _all_descriptors, int num_of_clusters);

    void lookUpWord(int _supervized, Mat &_in_descriptors, Mat &test_labels, Mat results, Mat entropy, Mat res_percent);

    float calculateVectorDistance(Mat& sourceVector, Mat& destinationVector);

};

#endif // BAGOFWORDS_H
