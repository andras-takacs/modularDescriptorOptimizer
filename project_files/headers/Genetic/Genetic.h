#ifndef GENETIC_H
#define GENETIC_H

#include "utils.h"

class DescriptorGeneration;

namespace Genetic
{

extern cv::Mat buildUpInfo;
float descriptorObjective(GAGenome& g);
void optimize();
void readScoreFile(std::vector<DescriptorGeneration> &_generations, int numInGen, int numOfGen);

};

#endif // GENETIC_H
