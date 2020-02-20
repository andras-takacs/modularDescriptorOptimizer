#include "MachineLearning/BagOfWords.h"

BagOfWords::BagOfWords()
{

}

BagOfWords::~BagOfWords()
{
    supervisedVocabulary.release();
    unSupervisedVocabulary.release();
    unsupervisedLabels.release();
}

void BagOfWords::calculateSupervisedVocabulary(Mat& _all_descriptors, Mat& _marked_labels){

    int d_cols = _all_descriptors.rows;
    double min=0, max_labels=0;
    cv::minMaxLoc(_marked_labels, &min, &max_labels);
    max_labels++;

    vector<Mat>words_sum((int)max_labels);
    vector<int>decriptor_couter((int)max_labels);
    vector<Mat>clustered_descriptors((int)max_labels);

    //!AVERAGING THE CLASS MEMEBER VECTORS
    for (int d_i=0;d_i<d_cols;++d_i){

        int row_id = (int)_marked_labels.at<float>(d_i,0);

        Mat summing_vector, av_summ_vec;
        summing_vector = words_sum.at(row_id);
        summing_vector.push_back(_all_descriptors.row(d_i));
        reduce(summing_vector,av_summ_vec,0,CV_REDUCE_SUM);
        decriptor_couter.at(row_id)++;
        words_sum[row_id] = av_summ_vec;

        summing_vector.release();
        av_summ_vec.release();

    }

    for(int voc_i=0;voc_i<max_labels;++voc_i){

        Mat averaging_vec;
        int total_desc;

        averaging_vec = words_sum.at(voc_i);
        total_desc = decriptor_couter.at(voc_i);

        Mat res_word = averaging_vec / total_desc;


        supervisedVocabulary.push_back(res_word);

        averaging_vec.release();
        res_word.release();
    }
    /*
    //!GET KMEANS FOR EACH CLASS
    for (int cl_i=0;cl_i<d_cols;++cl_i){

        int class_num = (int) _marked_labels.at<float>(cl_i,0);

        for(int dc_i=0;dc_i<max_labels;++dc_i){

            if(class_num==dc_i){

                clustered_descriptors[dc_i].push_back(_all_descriptors.row(cl_i));

            }else{

                continue;
            }

        }
    }

    for (int km_i=0;km_i<max_labels;++km_i){

        Mat centers, labels;

        cv::kmeans(clustered_descriptors[km_i], 1, labels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

        Mat av_center = supervisedVocabulary.row(km_i);
        float distance = calculateVectorDistance(centers,av_center);

        std::cout<<"Class "<<km_i<<" center: "<<centers<<std::endl;

        centers.release();
        labels.release();
    }

*/

}




void BagOfWords::calculateUnSupervisedVocabulary(Mat& _all_descriptors, int num_of_clusters){

    Mat centers, labels;

    cv::kmeans(_all_descriptors, num_of_clusters, labels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 15, KMEANS_PP_CENTERS, centers);

    std::cout<<"Vocabulary size: "<<centers.rows<<", labels size: "<<labels.rows<<std::endl;

    unSupervisedVocabulary=centers;
    unsupervisedLabels=labels;

    centers.release();
    labels.release();

}


void BagOfWords::lookUpWord(int _supervized, Mat& in_descriptors, Mat& test_labels, Mat results, Mat entropy, Mat res_percent){

    Mat dictionary;
    int num_bags = 0, num_descriptors = in_descriptors.rows;

    if(_supervized==SUPERVIZED){

        dictionary = supervisedVocabulary;

    }else if(_supervized==UNSUPERVIZED){

        dictionary = unSupervisedVocabulary;

    }

    num_bags = dictionary.rows;

    int false_positives[num_bags];
    int true_positives[num_bags];
    int total_class_result[num_bags];
    int total_good_results=0;

    vector<float>max_dists(num_bags);
    vector<float>min_dists(num_bags);
    vector<float>all_distances(num_descriptors);

    for(int nb=0; nb<num_bags; ++nb){
        false_positives[nb]=0;
        true_positives[nb]=0;
        total_class_result[nb]=0;

        min_dists[nb]=99999999999.9f;
        max_dists[nb]=0;
    };

    //    std::vector<std::vector< DMatch > >knn_matches;
    //    BFMatcher matcher(NORM_L2, true);
    //    std::vector<DMatch> good_matches;

    //    matcher.knnMatch(in_descriptors, dictionary , knn_matches,1);

    std::cout<<"Dictionary height: "<<dictionary.rows<<" width: "<<dictionary.cols<<std::endl;
    std::cout<<"Descriptors height: "<<in_descriptors.rows<<" width: "<<in_descriptors.cols<<std::endl;


    //!-- Quick calculation of max and min distances between keypoints
    for(int d_i = 0; d_i < num_descriptors;d_i++ )
    {
        float dist = 999999999999.9f;
        int result_idx = 0;

        for(int w_i=0;w_i<num_bags;++w_i){

            Mat sourceVector = in_descriptors.row(d_i);
            Mat destinationVector = dictionary.row(w_i);

            float temp_dist = calculateVectorDistance(sourceVector,destinationVector);

            if(temp_dist<dist){

                dist=temp_dist;
                result_idx = w_i;
            }
        }
        all_distances[d_i]=dist;
        //        std::cout<<"Result idx: "<<result_idx<<std::endl;

        //        dist = knn_matches[d_i][0].distance;
        results.at<float>(d_i,0)=(float)result_idx;


        for(int bd_i=0;bd_i<num_bags;++bd_i){

            if(bd_i==result_idx){

                if( dist < min_dists[bd_i] ) min_dists[bd_i] = dist;
                if( dist > max_dists[bd_i] ) max_dists[bd_i] = dist;

            }else{
                continue;
            }

        }


        if(result_idx == test_labels.at<float>(d_i,0)){
            total_good_results++;
            true_positives[result_idx]++;
        }else{
            false_positives[result_idx]++;
        }

    }

    for(int dt_i = 0; dt_i < num_descriptors; ++dt_i )
    {
        int result_idx = (int)results.at<float>(dt_i,0);
        float distance = all_distances[dt_i];
        entropy.at<float>(dt_i,0)=(float)(distance/max_dists[result_idx]);

    }


    for (int tcr_i = 0; tcr_i < num_bags; ++tcr_i)
    {
        total_class_result[tcr_i] = false_positives[tcr_i]+true_positives[tcr_i];
    }


    res_percent.at<float>(0,0)=(float)total_good_results/(float)num_descriptors;


    for (int rp_i = 0; rp_i < num_bags; ++rp_i)
    {

        res_percent.at<float>(0,rp_i+1)=(float) true_positives[rp_i]*100/total_class_result[rp_i];
        //        std::cout<<"True positive at class "<<i<<": "<<(double) true_positives[i]<<" Class results: "<<total_class_result[i]<<std::endl;
    }

}

float BagOfWords::calculateVectorDistance(Mat& sourceVector, Mat& destinationVector){

    float distance = 0;

    Mat distanceVector, squaredVector,summedVector;
    cv::subtract(destinationVector,sourceVector,distanceVector);
    cv::pow(distanceVector,2,squaredVector);
    cv::reduce(squaredVector,summedVector,1,CV_REDUCE_SUM);
    cv::sqrt(summedVector,summedVector);

    distance = summedVector.at<float>(0,0);

    //    std::cout<<"Distnace: "<<distance<<std::endl;

    distanceVector.release();
    squaredVector.release();
    summedVector.release();

    return distance;

}
