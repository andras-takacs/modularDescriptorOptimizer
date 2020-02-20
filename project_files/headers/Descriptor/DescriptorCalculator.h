#ifndef DESCRIPTORCALCULATOR_H
#define DESCRIPTORCALCULATOR_H

#include "utils.h"

using namespace std;
using namespace cv;

class DescriptorCalculator
{
public:
    DescriptorCalculator();
    DescriptorCalculator(std::shared_ptr<MDADescriptor> _descriptor);
    ~DescriptorCalculator();

    Mat calculate(Images& _in_im, vector<KeyPoint> const& _keypoint);
    Mat calculateOnePatch(ImagePatch& _in_patch);
    void concatVectors(vector<float>& _base_vector, vector<float>& _added_vector);
    Mat vectorToMat(vector<float>& _write_out_vector);

protected:
    Mat modularDescriptorResults;
    vector<DescriptorModule> descriptorModuleVector;
    std::shared_ptr<MDADescriptor> calculatedDescriptor;
};

#endif // DESCRIPTORCALCULATOR_H
