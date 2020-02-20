#ifndef DESCRIPTORASSEMBLER_H
#define DESCRIPTORASSEMBLER_H

#include "utils.h"

using namespace std;
using namespace cv;

class DescriptorModule;
class MDADescriptor;

class DescriptorAssembler
{
public:
    DescriptorAssembler();
    DescriptorAssembler(std::shared_ptr<MDADescriptor> _descriptor, int _projectFase=0);
    ~DescriptorAssembler();

    float descriptor_score;

    vector<DescriptorModule> getModuleVector(){return activeModules;}
    vector<int> getDecriptorGenome(){return descriptorGenome;}
    string writeOutAssebleGenome();
    int sumOfVector(vector<int>& _active_element);

    void buildUp();
    void loadValueToDescriptor(int _iterator, int _value);
    bool areActiveElements(std::vector<int>& _active_elem);

    int getSize(){return descriptorSize;}



private:
vector<DescriptorModule> descriptorSetup;
vector<DescriptorModule> activeModules;
vector<DescriptorModule> inactiveModules;
std::shared_ptr<MDADescriptor> assembleDescriptor;
//MDADescriptor* assembleDescriptor;
vector<int> descriptorGenome;
int descriptorSize;
int projectFase;

//!Training parameters
//!for Random Tree
int rTreeDepth; //level of splitting until reach the bottom of the tree max:20;
int numberOfTrees; //number of tree created in the forest, default=100;
int splitNumber; // number of variables randomly selected at node and used to find the best split(s) default=4;

//!For Preprocessing
int patchSize; //to set the patch size to see how big should be the optimal
int blurKernel; //gaussian blur kernel size for preprocessing
int blurSigma; //Sigma for Gaussian blur

};

#endif // DESCRIPTORASSEMBLER_H
