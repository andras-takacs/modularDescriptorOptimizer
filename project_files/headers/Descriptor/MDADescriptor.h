#ifndef MDADESCRIPTOR_H
#define MDADESCRIPTOR_H

#include "utils.h"

using namespace cv;
using namespace std;

enum COLORCHANNELS{RGB_CH,Lab_CH,Luv_CH,XYZ_CH,HSV_CH,HLS_CH,YCrCb_CH,OPP_CH};
enum KERNEL_SIZES{K_1,K_3,K_5,K_7,K_9,K_11,K_13,K_15};
enum PATCH_RADIUS{P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10};
enum RTREE_DEPTH{RTD_5,RTD_10,RTD_15,RTD_20,RTD_25,RTD_30,RTD_35,RTD_40};
enum NUMBER_OF_TREES{NT_40,NT_50,NT_60,NT_70,NT_80,NT_100,NT_120,NT_140};
enum SPLIT_NUMBER{SN_4,SN_8,SN_12,SN_16,SN_20,SN_25,SN_30,SN_35};
enum BLUR_SIGMA{SIG_1,SIG_2,SIG_3,SIG_4,SIG_5,SIG_6,SIG_7,SIG_8};
enum CANNY_KERNEL{CK_3,CK_5};
enum CANNY_THRESHOLD{CT_15,CT_25,CT_35,CT_45,CT_55,CT_65,CT_75,CT_85};
enum GRAD_DET{SOBEL,SHARR};
enum TOLERANCE{PCT_10,PCT_20,PCT_30,PCT_40,PCT_50,PCT_60,PCT_70,PCT_80};//2807 cambio NE
class DescriptorModule;

class MDADescriptor
{
public:
    MDADescriptor();
    MDADescriptor(vector<int> _genome);
    ~MDADescriptor();



//    void descriptorBuildUp(vector<int> _genome);

    /*Get and Set the Modular descriptor information*/
    void setModuleList(vector<DescriptorModule>& _module_list){module_list = _module_list;}
    vector<DescriptorModule> getModuleList(){return module_list;}
//    DescriptorModule getModule(int _position){return module_list[_position];}

    void setInactiveModuleList(vector<DescriptorModule>& _inactive_module_list){inactive_module_list = _inactive_module_list;}
    vector<DescriptorModule> getInactiveModuleList(){return inactive_module_list;}
//    DescriptorModule getInactiveModule(int _position){return inactive_module_list.at(_position);}

    void setGenome(vector<int>& _genome){descriptor_genome = _genome;}
    vector<int> getGenome(){return descriptor_genome;}

    void setFixGaussianKernelSize(int _kernel_size){gaussianKernelSize = _kernel_size;}
    int getGaussianKernelSize(){return gaussianKernelSize;}
    void setKernelFromGenome(int _k_enum);

    void setFixPatchRadius(int _radius){patchRadius = _radius;}
    int getPatchRadius(){return patchRadius;}
    void setPatchRadiusFromGenome(int _p_enum);

    void setFixTreeDepth(int _tree_depth){rTreeDepth = _tree_depth;}
    void setTreeDepthFromGenome(int _t_enum);
    int getTreeDepth(){return rTreeDepth;}

    void setFixTreeNumber(int _tree_number){numberOfTrees = _tree_number;}
    void setTreeNumberFromGenome(int _tn_enum);
    int getTreeNumbers(){return numberOfTrees;}

    void setFixSigma(int _sigma){blurSigma = _sigma;}
    void setSigmaFromGenome(int _s_enum);
    double getBlurSigma(){return blurSigma;}

    void setFixSplitNumber(int _splitNum){splitNumber =_splitNum;}
    void setSplitNumberFromGenome(int _sn_enum);
    int getSplitNumber(){return splitNumber;}

    void setColorChannelFromGenome(int _col_enum);
    string getColorName(){return colorChannelName;}
    int getColorChannel(){return usedColorChannel;}

    void setFixCannyKernelSize(int _ck_size){cannyKernelSize =_ck_size;}
    void setCannyKernelFromGenome(int _ck_enum);
    int getCannyKernelSize(){return cannyKernelSize;}

    void setFixCannyThreshold(int _ct_size){cannyThreshold =_ct_size;}
    void setCannyThresholdFromGenome(int _ct_enum);
    int getCannyThreshold(){return cannyThreshold;}

    void setDescriptorSize(int _size){descriptorSize = _size;}
    int getSize(){return descriptorSize;}

    void setGradientDetector(int _grad_enum);
    string getGradientDetectorName(){return gradientDetector;}
    int getGradientDetector(){return sobelOrSharr;}

    void setStringGenome(string _string_genome){stringGenome = _string_genome;}
    string getStringGenome(){return stringGenome;}


    string writeOutGenome(){return writeOutGenomeSequence(descriptor_genome);}
    string writeOutActivatorGenome(){return writeOutGenomeSequence(module_activator);}



    cv::Mat segmentationResults,calculationTimesResult;
    std::vector<double>segmentationScores, timeScores, invariantScores;
    std::vector<double> intensityShift,colorTemperature,intensityChange, blur,rotation,resize,affine;
    std::vector<double> lightChange,lightCondition1, lightCondition2, jpegCompression;

    float objectiveScore(){return _objectiveScore;}
    void setObjectiveScore(float _objSC){_objectiveScore = _objSC;}

    std::vector<int> moduleActivator(){return module_activator;}
    void moduleActivator(std::vector<int> m_act){module_activator=m_act;}

    int getModuleSizeTolerance(){return tolerance;}
    void setModuleSizeTolerance(int _tolerance){tolerance = _tolerance;}
    void setModuleSizeToleranceFromGenome(int _t_enum); //2807 cambio NE

    int getModuleListSize(){return module_list.size();}


private:



//!Training parameters
//!for Random Tree
int rTreeDepth; //level of splitting until reach the bottom of the tree max:20;
int numberOfTrees; //number of tree created in the forest, default=100;
int splitNumber; // number of variables randomly selected at node and used to find the best split(s) default=4;

//!For Preprocessing
int patchRadius;    //to set the patch size to see how big should be the optimal
int gaussianKernelSize; //gaussian blur kernel size for preprocessing
double blurSigma; //Sigma for Gaussian blur
int usedColorChannel;
int descriptorSize; //Length of the descriptor vector

int cannyKernelSize;
int cannyThreshold;
int sobelOrSharr;

int tolerance;//cambio

string colorChannelName;
string gradientDetector;
string stringGenome;

vector<int> descriptor_genome,module_activator;
vector<DescriptorModule> module_list;
vector<DescriptorModule> inactive_module_list;

float _objectiveScore;

string writeOutGenomeSequence(std::vector<int> _genome_vec);


};

#endif // MDADESCRIPTOR_H
