#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "utils.h"
using namespace cv;
using namespace cv::ml;


/*class CvRTreesMultiClass : public RTrees
{
public:
    float predict_multi_class( const CvMat* sample,
                               cv::AutoBuffer<int>& out_votes,
                               const CvMat* missing = 0) const;

    float predict_multi_class( const cv::Mat& sample,
                               cv::AutoBuffer<int>& out_votes,
                               const cv::Mat& missing) const;

};*/

class RandomForest
{
public:
    RandomForest();
    ~RandomForest();


    void loadTrainedForest(int i);

    void test(cv::Mat const &testing_data,
              cv::Mat const&testing_classifications,
              cv::Mat results,
              cv::Mat entropy,
              cv::Mat res_percents,
              const int &descriptor_type);

    void train(cv::Mat const&training_data,
               cv::Mat const&training_classifications,
               const int &descriptor_type, const int tree_depth=35, const int forest_size=100, const int number_for_split=16);

    static Ptr<TrainData> prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples);

    Ptr<RTrees> rtree;

};
#endif // RANDOMFOREST_H
