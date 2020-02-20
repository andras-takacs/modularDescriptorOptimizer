#ifndef EVALUATIONVALUES_H
#define EVALUATIONVALUES_H

#include "utils.h"

using namespace std;
using namespace cv;

class EvaluationValues
{
public:
    EvaluationValues();
    ~EvaluationValues();

    std::vector<double> getAlphas() {return alphas;}
    std::vector<std::vector<double>> getAlphaTriplets(){return alpha_triplets;}
    std::vector<double> getKelvins(){return kelvins;}
    std::vector<std::vector<double>> getAffineTriplets(){return affine_triplet;}
    std::vector<int> getRotations(){return rot_angles;}
    std::vector<double> getSizes(){return sizes;}
    std::vector<int> getBetas(){return betas;}
    std::vector<int> getKernelSizes(){return kernel_size;}

    double alphaAt(int _it) {return alphas[_it];}
    std::vector<double> alphaTripletAt(int _it){return alpha_triplets[_it];}
    double kelvinAt(int _it){return kelvins[_it];}
    std::vector<double> affineTripletAt(int _it){return affine_triplet[_it];}
    int rotationAt(int _it){return rot_angles[_it];}
    double sizeAt(int _it){return sizes[_it];}
    int betaAt(int _it){return betas[_it];}
    int kernelSizeAt(int _it){return kernel_size[_it];}

    int alphas_vec_length(){return alphas.size();}
    int alphaTriplets_vec_length(){return alpha_triplets.size();}
    int betas_vec_length(){return betas.size();}
    int kelvins_vec_length(){return kelvins.size();}
    int affine_vec_length(){return affine_triplet.size();}
    int rotations_vec_length(){return rot_angles.size();}
    int sizes_vec_length(){return sizes.size();}
    int kernels_vec_length(){return kernel_size.size();}

private:

    std::vector<double> alphas;  //alphas for the Light intesity change
    std::vector<std::vector<double>> alpha_triplets; // alpha triplets for the Light color change
    std::vector<int> betas; //betas for the Light intensity shift
    std::vector<double> kelvins; // kelvin values for the ploting
    std::vector<std::vector<double>> affine_triplet; //affine triplets for the distortion
    std::vector<int> rot_angles; //rotation values
    std::vector<double> sizes; //size multiplication values
    std::vector<int> kernel_size; //kernel size for the gaussian bluring

};

#endif // EVALUATIONVALUES_H
