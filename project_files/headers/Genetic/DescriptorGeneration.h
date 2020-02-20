#ifndef DESCRIPTORGENERATION_H
#define DESCRIPTORGENERATION_H

#include "utils.h"

using namespace cv;
using namespace std;

class DescriptorGeneration
{
public:
    DescriptorGeneration();
    ~DescriptorGeneration();

    std::vector<std::shared_ptr<MDADescriptor>> descriptorGeneration;
    std::vector<float> objective_scores;
    std::vector<float> fitness_scores;
    std::vector<string> genomes;
    std::vector<string> names;
    string best_genome;
    string best_name;
    float best_obejctive_score;
    float best_fitness_score;
    int best_genome_position;
    int generation;

    float averageObjScore, averageFitnessScore;

    void getBestIndividualPosition();
    void averageScores();
    void writeBestDescriptorInGeneration();
    string vectorIntToString(std::vector<int> _genome);
    string GetHexFromBin(string sBinary);
};

#endif // DESCRIPTORGENERATION_H
