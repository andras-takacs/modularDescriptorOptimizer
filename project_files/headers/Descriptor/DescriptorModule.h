#ifndef DESCRIPTORMODULE_H
#define DESCRIPTORMODULE_H

#include "utils.h"

using namespace std;
using namespace cv;

class ImagePatch;

class DescriptorModule
{
public:
    DescriptorModule();
    ~DescriptorModule();

    DescriptorModule(int& _module_id, vector<int>& _active_elements, int _size, int _module_position, int _project_fase=1);


//    void setModule(int &_module_id, int &_module_position, int &_size);

    void calculate(ImagePatch& _patch);


    //!Returning protected values
    int getID(){return module_id;}
    int size(){return module_size;}
    int defaultSize(){return default_module_size;}
    int position(){return module_position;}
    string name(){return module_name;}

    string writeOutModuleGenome();
    Mat descriptor_values;

    vector<float> getValues(){return module_values;}
    vector<int> getActiveElements(){return active_values;}
    double calculationTime(){return calculation_time;}
    bool isCalculated(){return calculated;}

protected:

    //!Calculate "BASE" modules
    void loadPosition(ImagePatch& _patch);
    void calculateAverageColorValues(ImagePatch& _patch);
    void calculateCentralMomentValues(ImagePatch& _patch);
    void loadDistanceTransformValues(ImagePatch& _patch);

    //!Calculate the "FINAL" modules
    void calculateMeanAndStdDevForPatch(ImagePatch& _patch);
    void calculate2ndCentralMoment(ImagePatch& _patch);
    void calculate3rdCentralMoment(ImagePatch& _patch);
    void calculate4thCentralMoment(ImagePatch& _patch);
    void calculate5thCentralMoment(ImagePatch& _patch);
    void calculateHuMoments(ImagePatch& _patch);
    void calculateGevers3L(ImagePatch& _patch);
    void calculateGevers3C(ImagePatch& _patch);
    void calculateGeusebroekHC(ImagePatch& _patch);
    void calculateAffineMoments(ImagePatch& _patch);
    void calculateEigenValues(ImagePatch& _patch);
    void calculateDTMeanAndSTD(ImagePatch& _patch);
    void calculateGradients(ImagePatch& _patch);


    string setModuleName(int &_module_id);
    int setDefaultModuleSize(int &_module_id);

    int module_id, default_module_size ,module_size, module_position, projectFase;
    vector<float> module_values;
    vector<int> active_values;
    double calculation_time;
    string module_name;
    bool calculated;
    bool isActive;

};

#endif // DESCRIPTORMODULE_H
