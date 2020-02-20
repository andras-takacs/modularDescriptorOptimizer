cmake_minimum_required(VERSION 3.14)
project(descriptor_optimizer)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fpermissive -std=c++0x -fPIC")
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui xfeatures2d features2d flann ml)
find_package(Qt5 COMPONENTS Core Gui Network Widgets PrintSupport)
find_package(OpenMP)

file(GLOB QT_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} resources/*.qrc)
file(GLOB QT_FORMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} Ploting/*.ui)

QT5_ADD_RESOURCES(QT_RESOURCES_CPP ${QT_RESOURCES})
QT5_WRAP_UI(QT_FORMS_HPP ${QT_FORMS})


include_directories(headers)
include_directories(/usr/local/include)
include_directories(headers/Ploting)

include_directories(
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Gui_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS}
        ${Qt5PrintSupport_INCLUDE_DIRS}
)

SET(GA_LIB /usr/local/lib/libga.a)

set(SOURCES
        main.cpp
        cpp/Dataset/ParsingCSV.cpp
        cpp/Images/Database.cpp
        cpp/Descriptor/Descriptor.cpp
        cpp/Segmentation/KeyInterestPoint.cpp
        cpp/Segmentation/ImageKeypoints.cpp
        cpp/Segmentation/ImageLabeling.cpp
        cpp/MachineLearning/RandomForest.cpp
        cpp/Segmentation/BoundingBox.cpp
        cpp/Images/Images.cpp
        cpp/Images/ImageProcessing.cpp
        cpp/Images/DescriptorImages.cpp
        cpp/Images/ImagePatch.cpp
        cpp/Evaluation/SegmentationEvaluation.cpp
        cpp/Ploting/qcustomplot.cpp
        cpp/Ploting/dialog.cpp
        cpp/Ploting/bardialog.cpp
        cpp/Ploting/postdialog.cpp
        cpp/Mathematics/Matek.cpp
        cpp/Mathematics/AffineMoments.cpp
        cpp/Mathematics/CentralMoments.cpp
        cpp/Mathematics/HuMomentInvariant.cpp
        cpp/Mathematics/MomentsCounter.cpp
        cpp/Mathematics/RawMoments.cpp
        cpp/Genetic/Genetic.cpp
        cpp/Descriptor/DescriptorAssembler.cpp
        cpp/Descriptor/DescriptorModule.cpp
        cpp/Descriptor/DescriptorCalculator.cpp
        cpp/Evaluation/GeneticEvaluation.cpp
        cpp/Descriptor/DescriptorExtract.cpp
        cpp/Evaluation/EvaluationValues.cpp
        cpp/MachineLearning/SparseCoding.cpp
        cpp/MachineLearning/BagOfWords.cpp
        cpp/MachineLearning/NaiveBayes.cpp
        cpp/MachineLearning/SupportVectorMachine.cpp
        cpp/MachineLearning/NeuralNetworks.cpp
        cpp/Descriptor/MDADescriptor.cpp
        cpp/Genetic/DescriptorGeneration.cpp
        cpp/Genetic/GADescriptorGA.cpp
        cpp/Evaluation/PostGeneticEvaluation.cpp
        cpp/Evaluation/PostGeneticBundleEvaluation.cpp
        )

set(QT_HEADERS
        headers/Ploting/qcustomplot.h
        headers/Ploting/dialog.h
        headers/Ploting/bardialog.h
        headers/Ploting/postdialog.h
        )

# Include directories for this project
set(INCLUDE_PATH
        ${OPENCV_INCLUDE_DIRS}
        )


# Libraries needed on all platforms for this project
set(LIBRARIES
        ${OpenCV_LIBS}
        ${GA_LIB}
        )

add_executable(descriptor_optimizer ${SOURCES} ${QT_HEADERS} ${QT_RESOURCES_CPP} ${QT_FORMS_HPP} ${QT_MOC_HPP})

# Use the Widgets module from Qt 5.
target_link_libraries(descriptor_optimizer PUBLIC OpenMP::OpenMP_CXX ${LIBRARIES} Qt5::Core Qt5::Widgets Qt5::PrintSupport Qt5::Gui )
