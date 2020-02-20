#ifndef GENETICEVALUATION_H
#define GENETICEVALUATION_H

#include "utils.h"
#include "Evaluation/EvaluationValues.h"

using namespace std;
using namespace cv;

class ImageKeypoints;

class GeneticEvaluation
{
public:
    GeneticEvaluation();
    GeneticEvaluation(std::shared_ptr<MDADescriptor> _descriptor, const int& _database, int _eval_im_number);
    ~GeneticEvaluation();

    void evaluate();
    float score(){return descriptor_global_score;} //return each evaluations objective score
    cv::Mat getMatchImages(){return img_matches;}   //return the matched images
    cv::Mat getTrasnformedImage(){return altered_im;} //return the transformed images
    std::vector<double>getScoreResults(){return score_results;} //return the score for each evaluation
    double scoreAt(int _it){return score_results[_it];} //return the score for evaluation _it
    void setComputer(int comp_id){usedComputer=comp_id;}

    double averageScore(); // calculates the global objective score from the evaluation scores
    int readNumbers( const string & s, vector <double> & v ); // function for to read homography matrix values
    void importMatrixFromFile(const char* filename_X, Mat _hom_mat, int& rows, int& cols); // read homography matrix files
    std::vector<cv::Mat> loadHomography(string folderRoute, int _image_size, const std::vector<string> _hom_list); // get homography matrix from files
    void testHomographyMatrix(vector<Mat> &homographies, vector<Images> &images); // test homography matrix on image pairs drawing outlines


    bool mark_keypoints; // boolian for marking keypoints in the evaluation phase

    // result vectors for ploting
    std::vector<double> intensityShiftResults,colorTemperatureResults,intensityChangeResults, blurResults,rotationResults,resizeResults,affineResults;
    std::vector<double> lightChangeResults,lightCondition1Results, lightCondition2Results, jpegCompressionResults;

    void cancelTransformation(){do_transformation_test=false;}
    void cancelTraining(){do_training_test=false;}

private:
    float descriptor_global_score;
    vector<DescriptorModule> assembledDescriptor; //assembled descriptor module vector
    std::shared_ptr<MDADescriptor> evaluatedDescriptor;

    //evaluation function values
    int usedDatabase, labelType, keyPointType, number_of_evaluation_images, matcher_type, comparation_method;
    int usedComputer;
    vector<double> score_results, weight;

    cv::Mat img_matches, altered_im;

    bool do_transformation_test, do_training_test;

    //evaluation functions
    void trainingEvaluation();
    void transformEvaluation();
    void transformEvaluation2();
    void oxfordEvaluation();
};

#endif // GENETICEVALUATION_H
