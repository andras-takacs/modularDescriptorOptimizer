#ifndef UTILS
#define UTILS

//!OPENCV HEADERS
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
//#include "opencv2/gpu/gpu.hpp"
//#include "opencv2/gpu/gpumat.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"

//!GLOBAL HEADERS
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <vector>
#include "boost/algorithm/string.hpp"

//!GALIB HEADERS
#include <ga/GA1DBinStrGenome.h>
#include <ga/GASimpleGA.h>

//!TILDE HEADERS
#include "KeypointDetector/3rdParties/tinydir.h"
#include "KeypointDetector/tilde/NonMaxSup.hpp"
#include "KeypointDetector/tilde/libTILDE.hpp"



//!Local Headers
//!
#include "Images/Database.h"
#include "Mathematics/Matek.h"
#include "Mathematics/RawMoments.h"
#include "Mathematics/CentralMoments.h"
#include "Mathematics/AffineMoments.h"
#include "Mathematics/HuMomentInvariant.h"
#include "Mathematics/MomentsCounter.h"
#include "Ploting/dialog.h"
#include "Ploting/bardialog.h"
#include "Ploting/qcustomplot.h"
#include "Ploting/postdialog.h"
#include "Images/Images.h"
#include "Images/ImageProcessing.h"
#include "Images/DescriptorImages.h"
#include "Images/ImagePatch.h"
#include "Descriptor/DescriptorModule.h"
#include "Descriptor/Descriptor.h"
#include "Descriptor/MDADescriptor.h"
#include "Genetic/DescriptorGeneration.h"
#include "Segmentation/KeyInterestPoint.h"
#include "Segmentation/ImageKeypoints.h"
#include "Segmentation/ImageLabeling.h"
#include "Segmentation/BoundingBox.h"
#include "Descriptor/DescriptorExtract.h"
#include "MachineLearning/RandomForest.h"
#include "MachineLearning/BagOfWords.h"
#include "MachineLearning/NaiveBayes.h"
#include "MachineLearning/SupportVectorMachine.h"
#include "MachineLearning/NeuralNetworks.h"
#include "Evaluation/SegmentationEvaluation.h"
#include "Descriptor/DescriptorAssembler.h"
#include "Descriptor/DescriptorCalculator.h"
#include "Evaluation/GeneticEvaluation.h"
#include "Evaluation/TestTilde.h"
#include "Genetic/GADescriptorGA.h"
#include "Genetic/Genetic.h"
#include "Evaluation/PostGeneticEvaluation.h"
#include "Evaluation/PostGeneticBundleEvaluation.h"




extern std::string homeDirectory;
extern uint NUMBER_OF_CLASSES;
extern uint number_of_test_images;
extern uint number_of_images;
extern bool trained;
extern uint descriptor_width;
//extern uint IM_WIDTH;
//extern uint IM_HEIGHT;

#define PI 3.14159265;

enum EVAL_TYPE {INTENSITY_TEMP, INTENSITY_CHANGE, INTENSITY_SHIFT, BLUR, ROTATE, SIZE, AFFINE, ILLUMINATION,TILDE_EVAL};
enum EVAL_TYPE2 {ROTATION_TR, AFFINE_TR, SIZE_TR, BLUR_TR, LIGHT_CH_TR,LIGHT_COND_TR, JPEG_COMPRESSION,LIGHT_COND_TR2};
enum MATCHER_TYPE {BRUTE_FORCE, FLANN, RAW_EUCLID, RAW_EUCLID_KNN};
enum COMPARATION_METHOD { DESCRIPTOR_INDEX, DOT_PRODUCT };

enum DESCRIPTOR_TYPE {SIFT_D, SURF_D, ORB_D, BRIEF_D, BRISK_D, FREAK_D, LATCH_D, M_PROJECT_D, D_PROJECT_D, RGB_D,  OPP_SIFT_D, OPP_SURF_D};
enum TEST_OR_TRAIN {TEST, TRAIN};
enum KEYPOINT_TYPE {RANDOM_KP, FAST_KP, ALL_PIX_KP, PATCH_DIST_KP};
enum LABEL_TYPE {PROJECT_2_CLASS, PROJECT_3_CLASS, PROJECT_4_CLASS, PROJECT_9_CLASS, LABELME_FACADE_9_CLASS };

//!DATABASE ENUMUERATIONS
enum COMPUTER_ID {LAPTOP, UNI_COMPUTER};
enum DATABASE {BRIGHTON_DB, LABELME_FACADE_DB_ALL, LABELME_FACADE_DB_JENA,TILDE_DB,OXFORD_DB};
enum OXFORD_SUBFOLDERS{BARK,BIKES,BOAT,LEUVEN,GRAF,NOTREDAME,OBAMA,PAINTEDLADIES,RUSHMORE,TREES,UBC,WALL,YOSEMITE};
enum TILDESUBFOLDERS{CHAMONIX,COURBEVOIE,FRANKFURT,MEXICO,PANORAMA,STLOUIS};

//!GENETIC ALGORITHM ENUMERATORS
enum DESCRIPTOR_BASE_MODULES {POSITION, AV_COLOR_VALUES, CENTRAL_MOMENTS, DISTANCE_TRANSFORM};
enum DESCRIPTOR_FINAL_MODULES {PATCH_POSITION, COLOR_MEAN_STD,CENTRAL_MOMENTS_2ND, CENTRAL_MOMENTS_3RD,
                               CENTRAL_MOMENTS_4TH,CENTRAL_MOMENTS_5TH, HU_MOMENTS,
                               GEVERS_3L,GEVERS_3C,GEUSEBROEK_H_C,AFFINE_MOMENTS,DT_MSTD,EIGEN_3,GRADIENT_CAL};
enum LEARNING_TYPE{SUPERVIZED, UNSUPERVIZED};
enum PROJECT_FASE{BASE,FINAL};


#endif // UTILS

