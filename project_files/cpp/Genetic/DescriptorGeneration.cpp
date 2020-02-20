#include "Genetic/DescriptorGeneration.h"

DescriptorGeneration::DescriptorGeneration()
{

}

DescriptorGeneration::~DescriptorGeneration()
{

}

void DescriptorGeneration::getBestIndividualPosition(){


    for(int g_it = 0;g_it<(int) genomes.size();++g_it){

        std::shared_ptr<MDADescriptor> curr_desc = descriptorGeneration[g_it];


        if (best_obejctive_score==curr_desc->objectiveScore()){

            best_genome_position = g_it;

            std::cout<<"In generation "<<generation<<" Best position is at the "<<best_genome_position<<" its score: "<<objective_scores[g_it]<<std::endl;

        }else{

            continue;
        }


    }

}

void DescriptorGeneration::averageScores(){


    float sumObj = std::accumulate(objective_scores.begin(), objective_scores.end(), 0.0);
    averageObjScore = sumObj / objective_scores.size();

    float sumFit = std::accumulate(fitness_scores.begin(), fitness_scores.end(), 0.0);
    averageFitnessScore = sumFit / objective_scores.size();


}

void DescriptorGeneration::writeBestDescriptorInGeneration(){

    ofstream outfile;
    string generations_val = "../genetic_results/Descriptor/allDescriptorsInGeneration"+std::to_string(generation)+".dat";

    outfile.open(generations_val.c_str());
    // Write information about all the individuals in each population to file and to class



    for(int d_it=0;d_it<(int)descriptorGeneration.size();++d_it){

        MDADescriptor bestDescriptor = *descriptorGeneration[d_it];
        vector<int>genome = bestDescriptor.getGenome();
        vector<DescriptorModule> bestModules = bestDescriptor.getModuleList();


        outfile<<"\nDescriptor no. "<<d_it<<"\n";
        std::stringstream ss_gen;
        for(int a=0; a<(int)genome.size(); a++){

            ss_gen << genome[a];
        }
        string ev_genome=ss_gen.str();
        outfile <<"Genome: "<<ev_genome<<"\n";
        outfile <<"Canny Kernel Size: "<<bestDescriptor.getCannyKernelSize()<<"\n";
        outfile <<"Canny min Threshold: "<<bestDescriptor.getCannyThreshold()<<"\n";
        outfile <<"Patch radius: "<<bestDescriptor.getPatchRadius()<<"\n";
        outfile <<"Gaussian Kernel Size: "<<bestDescriptor.getGaussianKernelSize()<<"\n";
        outfile <<"Blur Sigma: "<<bestDescriptor.getBlurSigma()<<"\n";
        outfile <<"Random Forest Size: "<<bestDescriptor.getTreeNumbers()<<"\n";
        outfile <<"Tree depth: "<<bestDescriptor.getTreeDepth()<<"\n";
        outfile <<"Split Number: "<<bestDescriptor.getSplitNumber()<<"\n";
        outfile <<"Color Model: "<<bestDescriptor.getColorName()<<"\n";
        outfile <<"Descriptor Size: "<<bestModules.size()<<"\n";
        outfile <<"Descriptor Modules:\n";
        for(int m_it = 0; m_it<(int)bestModules.size();++m_it){
            DescriptorModule curr_module = bestModules[m_it];

            outfile <<"Module name:"<<curr_module.name();
            outfile <<" and Genome: "<<curr_module.writeOutModuleGenome()<<"\n";
        }

        outfile <<"Descriptor's objective score: "<<bestDescriptor.objectiveScore() << "\n";
        outfile <<"Descriptor's Fitness score: "<<best_fitness_score << "\n\n";
        outfile <<"Segmentation percents: "<<bestDescriptor.segmentationResults<< "\n";
        outfile <<"Calculation time result: "<<bestDescriptor.calculationTimesResult<< "\n";

        outfile <<"Segmentation Scores: ";

        for(int i=0; i<(int)bestDescriptor.segmentationScores.size(); ++i){
            outfile << bestDescriptor.segmentationScores[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Invariant Scores: ";

        for(int i=0; i<(int)bestDescriptor.invariantScores.size(); ++i){
            outfile << bestDescriptor.invariantScores[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Time Scores: ";

        for(int i=0; i<(int)bestDescriptor.timeScores.size(); ++i){
            outfile << bestDescriptor.timeScores[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Light Change: ";

        for(int i=0; i<(int)bestDescriptor.lightChange.size(); ++i){
            outfile << bestDescriptor.lightChange[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Light Condition 1: ";

        for(int i=0; i<(int)bestDescriptor.lightCondition1.size(); ++i){
            outfile << bestDescriptor.lightCondition1[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Light Condition 2: ";

        for(int i=0; i<(int)bestDescriptor.lightCondition2.size(); ++i){
            outfile << bestDescriptor.lightCondition2[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Blur: ";

        for(int i=0; i<(int)bestDescriptor.blur.size(); ++i){
            outfile << bestDescriptor.blur[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Rotation: ";

        for(int i=0; i<(int)bestDescriptor.rotation.size(); ++i){
            outfile << bestDescriptor.rotation[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Resize: ";

        for(int i=0; i<(int)bestDescriptor.resize.size(); ++i){
            outfile << bestDescriptor.resize[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"Affine: ";

        for(int i=0; i<(int)bestDescriptor.affine.size(); ++i){
            outfile << bestDescriptor.affine[i] << ", ";
        }
        outfile <<"\n";

        outfile <<"JPEG compression: ";

        for(int i=0; i<(int)bestDescriptor.jpegCompression.size(); ++i){
            outfile << bestDescriptor.jpegCompression[i] << ", ";
        }
        outfile <<"\n";

    }
    outfile.close();
}

string DescriptorGeneration::vectorIntToString(std::vector<int> _genome){


    std::stringstream result;
    std::copy(_genome.begin(), _genome.end(), std::ostream_iterator<int>(result, ""));
    //    result.str().c_str();
    string outstring = result.str();

    return outstring;

}


string DescriptorGeneration::GetHexFromBin(string sBinary)
{
    string rest("0x"),tmp,chr = "0000";
    int len = sBinary.length()/4;
    chr = chr.substr(0,len);
    sBinary = chr+sBinary;
    for (int i=0;i<(int)sBinary.length();i+=4)
    {
        tmp = sBinary.substr(i,4);
        if (!tmp.compare("0000"))
        {
            rest = rest + "0";
        }
        else if (!tmp.compare("0001"))
        {
            rest = rest + "1";
        }
        else if (!tmp.compare("0010"))
        {
            rest = rest + "2";
        }
        else if (!tmp.compare("0011"))
        {
            rest = rest + "3";
        }
        else if (!tmp.compare("0100"))
        {
            rest = rest + "4";
        }
        else if (!tmp.compare("0101"))
        {
            rest = rest + "5";
        }
        else if (!tmp.compare("0110"))
        {
            rest = rest + "6";
        }
        else if (!tmp.compare("0111"))
        {
            rest = rest + "7";
        }
        else if (!tmp.compare("1000"))
        {
            rest = rest + "8";
        }
        else if (!tmp.compare("1001"))
        {
            rest = rest + "9";
        }
        else if (!tmp.compare("1010"))
        {
            rest = rest + "A";
        }
        else if (!tmp.compare("1011"))
        {
            rest = rest + "B";
        }
        else if (!tmp.compare("1100"))
        {
            rest = rest + "C";
        }
        else if (!tmp.compare("1101"))
        {
            rest = rest + "D";
        }
        else if (!tmp.compare("1110"))
        {
            rest = rest + "E";
        }
        else if (!tmp.compare("1111"))
        {
            rest = rest + "F";
        }
        else
        {
            continue;
        }
    }
    return rest;
}



