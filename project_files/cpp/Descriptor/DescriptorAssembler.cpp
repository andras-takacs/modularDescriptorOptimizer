#include "Descriptor/DescriptorAssembler.h"

DescriptorAssembler::DescriptorAssembler()
{
    //    assembleDescriptor = NULL;
}

DescriptorAssembler::DescriptorAssembler(std::shared_ptr<MDADescriptor> _descriptor, int _projectFase)
{
    assembleDescriptor = _descriptor;
    descriptorGenome = assembleDescriptor->getGenome();
    descriptorSize=sumOfVector(descriptorGenome);
    projectFase = _projectFase;
}

DescriptorAssembler::~DescriptorAssembler()
{

    //    if(assembleDescriptor != NULL){
    //        delete assembleDescriptor;
    //    }
}

void DescriptorAssembler::buildUp(){

    //! The end position of the descriptor modules
    std::vector<uint> module_evaluation_point;
    std::vector<uint> module_size_list;

    if(projectFase==BASE){
        module_evaluation_point.reserve(4);
        module_evaluation_point = {1,7,31,112};
        module_size_list.reserve(4);
        module_size_list = {2,6,24,81};

    }else if(projectFase==FINAL){

        module_evaluation_point.reserve(18);
        module_evaluation_point = {2,5,8,11,14,17,20,22,28,37,49,64,82,106,112,118,122,152};
        module_size_list.reserve(25);
        module_size_list = {3, // 1 - Color channel
                            3, // 2 - Gaussian kernel size
                            3, // 3 - Patch Radius size
                            3, // 4 - Random Tree Depth
                            3, // 5 - Number Of Trees in Random Forest
                            3, // 6 - Number of variables randomly selected for best split
                            3, // 7 - Blur Sigma
                            1, // 8 - Canny Kernel
                            3, // 9 - Canny low threshold
                            1, // 10- Sobel or Sharr decider
                            3, // 11- tolerance value  2807 cambio NE
                            //=============== From here is value activation ========================
                            2, // 1 - Patch center position
                            6, // 2 - Mean and Standard deviation(σ) for each channel
                            9, // 3 - 2nd Order Normalized Central moment (nu_20,nu_11,nu_02)
                            12,// 4 - 3rd Order Normalized Central moment (nu_30,nu_21,nu_12,nu_03)
                            15,// 5 - 4th Order Normalized Central moment (nu_40,nu_31,nu_22,nu_13,nu_04)
                            18,// 6 - 5th Order Normalized Central moment (nu_50,nu_41,nu_32,nu_23,nu_14,nu_05)
                            24,// 7 - Hu moments(8) for each channels
                            6, // 8 - Mean and standard deviation of Gavers l1,l2,l3 features
                            6, // 9 - Mean and standard deviation of Gavers c1,c2,c3 features
                            4, // 10- Mean and standard deviation of Geusebroek H and C features
                            30,// 11- Affine Moments
                            2, // 12- Meand and Std Deviation of Distance Transform
                            9, // 13- Eigen Values (3 highest from each channel)
                            45 // 14- Gradient results (Average of gradient angles 15, Average Angle differences 30)
                           };
    }

    /* This vector contains the active descriptor elements
     and will be loaded for the construction of Descriptor Module*/
    std::vector<int> _active_elem;

    /* Counting how many modules has passed,
     * and gives the id for the DescriptorModule
     * The first 7 binary module turned into number:
     * 1 - Color channel (3 bits) ==> 8 channels
     * 2 - Gaussian kernel size (3 bits) ==> max 15 pix
     * 3 - Patch Radius size (3 bits) ==> max radius 8 pix
     * 4 - Random Tree Depth (3 bits) ==> max depth 40
     * 5 - Number Of Trees in the Forest (3 bits) ==> max trees 140
     * 6 - Number of variables randomly selected for best split (3 bits) ==> max var 35
     * 7 - Blur Sigma (3 bits) ==> mas sigma 35
     * 8 - Canny Kernel
     * 9 - Canny low threshold
     * 10- Sobel or Sharr
     * 11- Tolerance (3 bits) ==> range (10-80)
*/
    int number_of_preproc_modules = 11;

    int module_iterator = 0;
    int position_counter = 0;
    int eval_point = 0;

    if(projectFase==BASE){
        eval_point = 1;
    }else if(projectFase==FINAL){
        eval_point = 2;
    }

    int real_descriptor_size = 0;

    //crear funcion porcentaje - J.C.
    //!The ModularDescriptor creation loop
    for (uint t_it=0;t_it<descriptorGenome.size();++t_it){

        _active_elem.push_back(descriptorGenome.at(t_it));

        //        std::cout<<"Eval Point: "<<eval_point<<std::endl;
        //        if(t_it==module_evaluation_point.at((uint) module_iterator)){
        if(t_it==(uint) eval_point){

            //!The first 11 module are converted to number
            if(module_iterator<number_of_preproc_modules){

                int binValue = Matek::binVectorToInt(_active_elem);
                int moduleValue = Matek::binaryToDecimal(binValue);
                loadValueToDescriptor(module_iterator,moduleValue);

                _active_elem.clear();
                module_iterator++;
                eval_point+=module_size_list[module_iterator];
            }else{

                int function_module_it = module_iterator-number_of_preproc_modules;
                //!If there is at least 1 active element, the module is created
                std::cout<<"Iteration: "<<t_it<<std::endl;

                if(areActiveElements(_active_elem)){
                    int module_real_size = sumOfVector(_active_elem);
                    real_descriptor_size+=module_real_size;
                    DescriptorModule module(function_module_it,_active_elem,module_real_size, position_counter,projectFase);
                    activeModules.push_back(module);
                    std::cout<<"Active module: "<<module.name()<<std::endl;
                    //!The vector is cleaned for the next module
                    _active_elem.clear();
                    module_iterator++;int binValue = Matek::binVectorToInt(_active_elem);
                    int moduleValue = Matek::binaryToDecimal(binValue);
                    position_counter++;
                    eval_point+=module_size_list[module_iterator];
                }else if (!areActiveElements(_active_elem)){
                    DescriptorModule inactive_module(function_module_it,_active_elem,0, 0,projectFase);
                    inactiveModules.push_back(inactive_module);
                    std::cout<<"Inactive module: "<<inactive_module.name()<<std::endl;
                    _active_elem.clear();
                    module_iterator++;
                    eval_point+=module_size_list[module_iterator];
                }
            }

        }else{
            continue;
        }
    }
    descriptorSize = real_descriptor_size;
    assembleDescriptor->setModuleList(activeModules);
    assembleDescriptor->setInactiveModuleList(inactiveModules);
    assembleDescriptor->setDescriptorSize(descriptorSize);
}

//!if the sum of active_elements are 0, then there is false --> no active element
bool DescriptorAssembler::areActiveElements(std::vector<int>& _active_elem){ //cambio

    activeElements = true;
    double porcentajeInactivo=0;
    porcentajeInactivo = 100 - ( ( ((double) sumOfVector(_active_elem)) / ((double) _active_elem.size()) )*100.0);
    cout<<"Porcentaje inactivo: "<<porcentajeInactivo<<endl;
    cout<<"tolerancia: "<<assembleDescriptor->getModuleSizeTolerance()<<endl;
    if (porcentajeInactivo>=assembleDescriptor->getModuleSizeTolerance()){
        activeElements = false;
        //cout<<"FAIL"<<endl;
    }else{
        activeElements = true;
        //cout<<"PASS"<<endl;
    }

    return activeElements;
}

string DescriptorAssembler::writeOutAssebleGenome(){

    stringstream ss_genom;
    std::copy( descriptorGenome.begin(), descriptorGenome.end(), ostream_iterator<int>(ss_genom, ""));
    string string_out = ss_genom.str();
    string_out = string_out.substr(0, string_out.length());

    return string_out;
}

int DescriptorAssembler::sumOfVector(vector<int> &_active_element){

    int sum_of_elems=0;

    sum_of_elems =std::accumulate(_active_element.begin(),_active_element.end(),0);

    return sum_of_elems;
}

void DescriptorAssembler::loadValueToDescriptor(int _iterator, int _value){

    /* 1 - Color channel (3 bits) ==> 8 channels
    *  2 - Gaussian kernel size (3 bits) ==> max 15 pix
    *  3 - Patch Radius size (3 bits) ==> max radius 8 pix
    *  4 - Random Tree Depth (3 bits) ==> max depth 40
    *  5 - Number Of Trees in the Forest (3 bits) ==> max trees 140
    *  6 - Number of variables randomly selected for best split (3 bits) ==> max var 35
    *  7 - Blur Sigma (3 bits) ==> max sigma 30
    *  8 - Canny Kernel size (1 bits) ==> (3,5)
    *  9 - Canny low threshold (3 bits) ==> highest min threshold 85 --> for to have 85*3=255 the high threshold
    *  10- Define the gradient detector ===> Sobel or Sharr
    */

    //    std::cout<<"Module value: "<<_value<<std::endl;


    switch (_iterator)
    {
    case 0:
        assembleDescriptor->setColorChannelFromGenome(_value);
        break;
    case 1:
        assembleDescriptor->setKernelFromGenome(_value);
        break;
    case 2:
        assembleDescriptor->setPatchRadiusFromGenome(_value);
        break;
    case 3:
        assembleDescriptor->setTreeDepthFromGenome(_value);
        break;
    case 4:
        assembleDescriptor->setTreeNumberFromGenome(_value);
        break;
    case 5:
        assembleDescriptor->setSplitNumberFromGenome(_value);
        break;
    case 6:
        assembleDescriptor->setSigmaFromGenome(_value);
        break;
    case 7:
        assembleDescriptor->setCannyKernelFromGenome(_value);
        break;
    case 8:
        assembleDescriptor->setCannyThresholdFromGenome(_value);
        break;
    case 9:
        assembleDescriptor->setGradientDetector(_value);
        break;
    case 10:
        assembleDescriptor->setModuleSizeToleranceFromGenome(_value);
    default:

        break;
    }
}
