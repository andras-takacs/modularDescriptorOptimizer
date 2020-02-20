#include "Genetic/Genetic.h"

namespace Genetic{
cv::Mat buildUpInfo;
std::vector< std::shared_ptr<MDADescriptor>>descGeneration;

const int _genome_length = 217;//2807cambio 228 a 231
const int _pop_size =50;
bool resetOld = false;
}

float Genetic::descriptorObjective(GAGenome& g)
{

    std::vector<int>bin_genome_vec;
    const GA1DBinaryStringGenome * const genome
            = dynamic_cast<GA1DBinaryStringGenome*>(&g);
    assert(genome && "Assume a binary string genome for triplet objective");
    const int maxx = genome->length();



    float score=0.0;

    std::ostringstream ss_gen;
    //    static char syms[] = "0123456789ABCDEF";
    //    ss_gen << std::hex;
    for(int a=0; a<maxx; a++){

        //        ss_gen << std::hex << std::setfill('0') << std::setw(0) << (int)genome->gene(a);
        ss_gen << std::hex << std::setw(0) << (unsigned short)genome->gene(a);
    }
    string str_genome=ss_gen.str();
    //    std::stringstream ss_hex;
    //    ss_hex << std::showbase << std::uppercase << std::hex << str_genome;
    //    str_genome = ss_hex.str();
    //    str_genome.erase(str_genome.find_last_not_of(" \f\n\r\t\v") + 1);
    //    str_genome.find_first_not_of( " \f\n\r\t\v" );


    for(int x=0; x<maxx; ++x)
    {
        //        if(x<2){
        //            bin_genome_vec.push_back(0);
        //        }else{
        bin_genome_vec.push_back(genome->gene(x));
        //        }

    }

    //std::cout<<"Got here!"<<std::endl;

    std::shared_ptr<MDADescriptor> mDescriptor (new MDADescriptor());

    mDescriptor->setGenome(bin_genome_vec);
    mDescriptor->setStringGenome(str_genome);
    mDescriptor->setModuleSizeTolerance(0);//cambio
    //    MDADescriptor mDescriptor(bin_genome_vec);

    DescriptorAssembler modularDescriptor(mDescriptor,FINAL);

    modularDescriptor.buildUp();

    if(!modularDescriptor.isThereActiveElements()){
        score = 200;
    }else{

        GeneticEvaluation genomeDescriptorEvaluation(mDescriptor, BRIGHTON_DB,3);
        genomeDescriptorEvaluation.setComputer(LAPTOP);
        //genomeDescriptorEvaluation.cancelTraining();


        //    double  sum_of_elems = std::accumulate(score_vector.begin(), score_vector.end(), 0.0);

        //    score = (float) (sum_of_elems/(double) score_vector.size());


        std::cout<<"Descriptor length: "<<mDescriptor->getSize()<<"\n"
                <<"Patch size: "<<mDescriptor->getPatchRadius()<<"\n"
               <<"Kernel Size: "<<mDescriptor->getGaussianKernelSize()<<"\n"
              <<"Canny threshold: "<<mDescriptor->getCannyThreshold()<<"\n"
             <<"Canny Kernel Size: "<<mDescriptor->getCannyKernelSize()<<"\n"
            <<"Color Channel: "<<mDescriptor->getColorName()<<std::endl;


        std::cout<<"Assemble Genome: "<<mDescriptor->writeOutGenome()<<std::endl;
        std::cout<<"Activator Genome: "<<mDescriptor->writeOutActivatorGenome()<<std::endl;
        std::cout<<"Size of the whole descriptor: "<<mDescriptor->getSize()<<" active values"<<std::endl;

        std::cout<<"Modular Descriptor setup: "<<mDescriptor->getModuleList().size()<<" active modules"<<std::endl;
        for (uint i=0;i<mDescriptor->getModuleList().size();++i){
            std::cout<<i<<": "<<mDescriptor->getModuleList().at(i).name()<<"\n"
                    <<"Optimized size: "<<mDescriptor->getModuleList().at(i).size()<<"\n"
                   <<"Original size:"<<mDescriptor->getModuleList().at(i).defaultSize()<<"\n"
                  <<"Genome: "<<mDescriptor->getModuleList().at(i).writeOutModuleGenome()<<std::endl;

        }

        int desc_size = mDescriptor->getSize();
        float rel_size = (float) desc_size/(float) maxx;
        //    std::cout<<"Relative size: "<<rel_size<<"% of the whole descriptor"<<std::endl;

        genomeDescriptorEvaluation.evaluate();

        std::vector<double> score_vector = genomeDescriptorEvaluation.getScoreResults();

        std::cout<<"Size of score vector: "<<score_vector.size()<<std::endl;

        //score = (float) (std::accumulate(score_vector.begin(), score_vector.end(), 0.0));

        for(int s_i=0;s_i<score_vector.size();s_i++){

            float actual_score = 0;
            if(score_vector[s_i]!=score_vector[s_i]){
                actual_score = 1;
            }else {
                actual_score = score_vector[s_i];
            }
            score+=actual_score;
        }

        score+=rel_size;

        //    std::cout<<"Genome score: "<<score<<std::endl;

        mDescriptor->setObjectiveScore(score);
        //IF THERE IS A PROBLEM WITH THE SCORE, PRINT ALL RESULTS INTO A FILE
        if(score!=score){
            ofstream prob_gen;
            string problematic_file_address = "../genetic_results/Genetic/problematic_individual.dat";
            prob_gen.open(problematic_file_address.c_str());
            for(int pr_s=0;pr_s < score_vector.size();pr_s++){
                prob_gen << "Score "<<pr_s<<": "<<score_vector[pr_s]<<"\n";
            }
            prob_gen.close();
        }

        bin_genome_vec.clear();

        //    score = modularDescriptor.descriptor_score;
    }
    descGeneration.push_back(mDescriptor);

    return score;
}





void Genetic::optimize(){


    GAParameterList params;
    GADescriptorGA::registerDefaultParameters(params);

    params.set(gaNnReplacement, GAPopulation::WORST);
    params.set(gaNpReplacement, 0.8);


    //Create first genome
    const int genome_length = _genome_length;
    //    const int ploidy = 2; //Diploid

    // typedef float (*Evaluator)(GAGenome &);
    //    Genetic::buildUpInfo = cv::Mat(1,genome_length,CV_32FC1,cv::Scalar(13));
    //    cv::randu(buildUpInfo, Scalar::all(0), Scalar::all(255));
    const GA1DBinaryStringGenome first_genome(genome_length, Genetic::descriptorObjective);
    int maxCoresUsed = 7;

    //Set up the genetic algorithm
    GADescriptorGA g(first_genome,maxCoresUsed);

    g.minimize();
    g.populationSize(_pop_size);
    g.nGenerations(1000);

    g.pMutation(0.01);  // likelihood of mutating new offspring
    g.pCrossover(0.8);   // likelihood of crossing over parents
    g.scoreFilename("generations.dat"); // name of output file for scores
    g.scoreFrequency(1);       // keep the scores of every generation
    g.flushFrequency(1);           // specify how often to write the score to disk



    g.selectScores(GAStatistics::AllScores); /* writing out average, maximum,
                                                  minimum, standard deviation and
                                                  divergence values of the fitness
                                                  function for later plotting. */


    std::cout<<"Evolution started..."<<std::endl;
    //        g.initialize();
    //        for(int i=0; i<g.nGenerations();++i){
    //            std::cout<<"GENERATION NO: "<<i<<"."<<std::endl;
    //            g.step();
    //            //        std::cout<<"Generation no "<<g.statistics().generation()<<"."<<std::endl;
    //                    std::cout<<"Best Genome: no "<<g.statistics().bestIndividual()<<std::endl;
    //        }




    std::vector<DescriptorGeneration> evolution;

    g.initialize();


    if(resetOld){

        std::cout<<"Reseting"<<std::endl;



        string _filename = "../genetic_results/Genetic/geneOrigin.dat";

        string line;
        int it=0;
        ifstream myfile (_filename);
        if (myfile.is_open())
        {

            float _score = 0.0;
            float _fitness = 0.0;
            string _name ="";

            while ( getline (myfile,line) )
            {
                std::istringstream iss(line);
                GAGenome& indiv = g.population().individual(it);
                const GA1DBinaryStringGenome * const genome
                        = dynamic_cast<GA1DBinaryStringGenome*>(&indiv);

                //                std::cout<<"Before: "<<*genome<<std::endl;

                genome->read(iss);
                iss >> _score >> _fitness >> _name;



                indiv.fitness(_fitness);
                indiv.score(_score);


                //                std::cout<<"Score: "<< indiv.score()<<" Fitness: "<<indiv.fitness()<<" Name: "<<_name<<std::endl;
                //                std::cout<<"After: "<<*genome<<std::endl;

                it++;
            }
        }

        //        cv::waitKey();

    }


    ofstream outfile,g_outfile;

    //    string best_in_geneartion = "../genetic_results/bestInGenerations.dat";

    //    outfile_b.open(best_in_geneartion.c_str());

    //    while(!g.done()){
    for(int i=0; i<g.nGenerations();++i){

        std::cout<<"GENERATION NO: "<<g.statistics().generation()<<"."<<std::endl;
        std::cout<<"Best Genome: no "<<g.statistics().bestIndividual()<<std::endl;

        DescriptorGeneration curr_gen;
        curr_gen.generation = g.statistics().generation();
        string generations_val = "../genetic_results/Genetic/generation"+std::to_string(g.statistics().generation())+".dat";
        string gene_strings = "../genetic_results/Genetic/genepopulation"+std::to_string(g.statistics().generation())+".dat";
        outfile.open(generations_val.c_str());
        g_outfile.open(gene_strings.c_str());
        // Write information about all the individuals in each population to file and to class
        for(int go_i=0;go_i<g.population().size();++go_i){


            GAGenome& indiv = g.population().individual(go_i);
            const GA1DBinaryStringGenome * const genome
                    = dynamic_cast<GA1DBinaryStringGenome*>(&indiv);

            std::stringstream ss_gen;
            for(int a=0; a<genome_length; a++){
                g_outfile << genome->gene(a);
                outfile << genome->gene(a);
                ss_gen << genome->gene(a);
            }
            string ev_genome=ss_gen.str();
            //            ev_genome.erase(ev_genome.find_last_not_of(" \f\n\r\t\v") + 1);
            //            boost::algorithm::trim(ev_genome);
            curr_gen.genomes.push_back(ev_genome);
            curr_gen.fitness_scores.push_back(indiv.fitness());
            curr_gen.objective_scores.push_back(indiv.score());
            string e_name = "Individual"+std::to_string(go_i);
            curr_gen.names.push_back(e_name);
            g_outfile << "\n";
            outfile << " ";
            outfile << indiv.score() << " ";
            outfile << " ";
            outfile << indiv.fitness() << " Individual"<<go_i<<"\n";
        }

        // Write information about the best individual in each population to file and to class
        GAGenome& indiv_b = g.population().best();



        const GA1DBinaryStringGenome * const b_genome
                = dynamic_cast<GA1DBinaryStringGenome*>(&indiv_b);

        std::stringstream ss_gen_b;
        ss_gen_b << std::hex;
        for(int a=0; a<genome_length; a++){

            outfile << b_genome->gene(a);
            ss_gen_b << b_genome->gene(a);
        }
        string ev_genome_b=ss_gen_b.str();
        //        std::stringstream ss_hex;
        //        ss_hex << std::showbase << std::uppercase << std::hex << ev_genome_b;
        //        ev_genome_b = ss_hex.str();
        //        ev_genome_b.erase(ev_genome_b.find_last_not_of(" \f\n\r\t\v") + 1);
        //        ev_genome_b.find_first_not_of( " \f\n\r\t\v" );
        //        boost::algorithm::trim(ev_genome_b);
        curr_gen.best_genome = ev_genome_b;
        curr_gen.best_fitness_score = indiv_b.fitness();
        curr_gen.best_obejctive_score = indiv_b.score();
        string e_name = "Best genome";
        curr_gen.names.push_back(e_name);

        outfile << " ";
        outfile << indiv_b.score() << " ";
        outfile << " ";
        outfile << indiv_b.fitness() <<" Best genome "<<"\n";



        outfile.close();
        g_outfile.close();

        //Evaluate the Generation


        curr_gen.descriptorGeneration = descGeneration;
        descGeneration.clear();

        curr_gen.getBestIndividualPosition();
        curr_gen.writeBestDescriptorInGeneration();
        evolution.push_back(curr_gen);

        g.step();
        //        ++g;
    }
    g.flushScores();


    //    outfile_b.close();

    std::cout<<"Generations: "<<g.populationSize()<<std::endl;

    //    readScoreFile(evolution,g.populationSize(),g.nGenerations());


    for(int e_it=0;e_it<(int) evolution.size();++e_it){

        evolution[e_it].getBestIndividualPosition();

    }

    std::cout<<"Evolution size: "<<evolution.size()<<std::endl;


}

void Genetic::readScoreFile(std::vector<DescriptorGeneration>& _generations, int numInGen, int numOfGen){


    for(int gen_it = 0;gen_it<numOfGen;++gen_it){

        string _genome;
        string _name;
        float _score;
        float _fitness;

        std::vector<string>_genomes(numInGen);
        std::vector<string>_names(numInGen);
        std::vector<float>_scores(numInGen);
        std::vector<float>_fitnesses(numInGen);

        int pop_it = 0;
        string _filename = "../genetic_results/Genetic/generation"+std::to_string(gen_it)+".dat";

        string line;
        ifstream myfile (_filename);
        if (myfile.is_open())
        {
            while ( getline (myfile,line) )
            {
                std::istringstream iss(line);
                iss >> _genome >> _score >> _fitness >> _name;

                //                std::cout<<"Finess of "<<_name<<": "<<_score<<std::endl;

                if(pop_it!=numInGen){
                    _genomes.at(pop_it)=_genome;
                    _scores.at(pop_it)=_score;
                    _fitnesses.at(pop_it)=_fitness;
                    _names.at(pop_it) = _name;
                }

                if(pop_it==numInGen){

                    //                    std::cout<<"Pop it: "<<pop_it<<std::endl;
                    _generations[gen_it].genomes = _genomes;
                    _generations[gen_it].objective_scores = _scores;
                    _generations[gen_it].fitness_scores = _fitnesses;
                    _generations[gen_it].names = _names;


                    _generations[gen_it].best_genome = _genome;
                    _generations[gen_it].best_obejctive_score = _score;
                    _generations[gen_it].best_fitness_score = _fitness;
                    _generations[gen_it].best_name = _name;

                }



                pop_it++;

            }
            myfile.close();
        }
        else {
            cout << "Unable to open file";
        }
    }
}
