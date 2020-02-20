#ifndef POSTGENETICEVALUATION_H
#define POSTGENETICEVALUATION_H

#include "utils.h"

class PostGeneticEvaluation
{
public:
    PostGeneticEvaluation();
    PostGeneticEvaluation(const int &_database, int _eval_im_number=5, int setComputer=1);
    ~PostGeneticEvaluation();
    void evaluate();
    void calculateModularDescriptor();
    void plotResults();


    void genome(std::vector<int>_genome){modularGenome=_genome;}
    std::vector<int>genome(){return modularGenome;}


    double averageScore(); // calculates the global objective score from the evaluation scores
    int readNumbers( const string & s, vector <double> & v ); // function for to read homography matrix values
    void importMatrixFromFile(const char* filename_X, Mat _hom_mat, int& rows, int& cols); // read homography matrix files
    std::vector<cv::Mat> loadHomography(string folderRoute, int _image_size, const std::vector<string> _hom_list); // get homography matrix from files

    std::vector<std::vector<double>> blurResults,rotationResults,resizeResults,affineResults;
    std::vector<std::vector<double>> lightChangeResults,lightCondition1Results, lightCondition2Results, jpegCompressionResults;
    std::vector<double>times;

private:
    std::vector<int>modularGenome;
    int number_of_evaluation_images,usedDatabase,matcher_type,comparation_method,usedComputer;
    bool mark_keypoints;

    std::shared_ptr<MDADescriptor> evaluatedDescriptor;
    vector<double> score_results, weight;

    void trainingEvaluation(int desc_round);
    void transformEvaluation2(int desc_round);
    void testHomographyMatrix(vector<Mat>& homographies, vector<Images>& images);
    void oxfordEvaluation();
    void colorAndSaveClassResults(vector<KeyPoint> _keypoints, cv::Mat _image, cv::Mat _results, cv::Mat _entrophy, int _desc_round, int _im_turn);
    string getDescriptorName(int _desc_round);

};

#endif // POSTGENETICEVALUATION_H
