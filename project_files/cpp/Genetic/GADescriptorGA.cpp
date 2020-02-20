#include "Genetic/GADescriptorGA.h"
#include <ga/garandom.h>


GAParameterList&
GADescriptorGA::registerDefaultParameters(GAParameterList& p) {
  GAGeneticAlgorithm::registerDefaultParameters(p);

  p.add(gaNelitism, gaSNelitism,
    GAParameter::BOOLEAN, &gaDefElitism);

  return p;
}

GADescriptorGA::GADescriptorGA(const GAGenome& c, const int & _maxCoresUsed) : GAGeneticAlgorithm(c){
  oldPop = pop->clone();
    maxCoresUsed = _maxCoresUsed;
  el = gaTrue;
  params.add(gaNelitism, gaSNelitism, GAParameter::BOOLEAN, &el);
}
GADescriptorGA::GADescriptorGA(const GAPopulation& p) : GAGeneticAlgorithm(p){
  oldPop = pop->clone();

  el = gaTrue;
  params.add(gaNelitism, gaSNelitism, GAParameter::BOOLEAN, &el);
}
GADescriptorGA::GADescriptorGA(const GADescriptorGA& ga) : GAGeneticAlgorithm(ga){
  oldPop = (GAPopulation *)0;
  copy(ga);
}
GADescriptorGA::~GADescriptorGA(){
  delete oldPop;
}
GADescriptorGA&
GADescriptorGA::operator=(const GADescriptorGA& ga){
  if(&ga != this) copy(ga);
  return *this;
}
void
GADescriptorGA::copy(const GAGeneticAlgorithm & g){
  GAGeneticAlgorithm::copy(g);
  const GADescriptorGA& ga = DYN_CAST(const GADescriptorGA&,g);
  el = ga.el;
  if(oldPop) oldPop->copy(*(ga.oldPop));
  else oldPop = ga.oldPop->clone();
  oldPop->geneticAlgorithm(*this);
}


int
GADescriptorGA::setptr(const char* name, const void* value){
  int status = GAGeneticAlgorithm::setptr(name, value);

  if(strcmp(name, gaNelitism) == 0 ||
     strcmp(name, gaSNelitism) == 0){
    el = (*((int*)value) != 0 ? gaTrue : gaFalse);
    status = 0;
  }
  return status;
}

int
GADescriptorGA::get(const char* name, void* value) const {
  int status = GAGeneticAlgorithm::get(name, value);

  if(strcmp(name, gaNelitism) == 0 ||
     strcmp(name, gaSNelitism) == 0){
    *((int*)value) = (el == gaTrue ? 1 : 0);
    status = 0;
  }
  return status;
}

void
GADescriptorGA::objectiveFunction(GAGenome::Evaluator f){
  GAGeneticAlgorithm::objectiveFunction(f);
  for(int i=0; i<pop->size(); i++)
    oldPop->individual(i).evaluator(f);
}

void
GADescriptorGA::objectiveData(const GAEvalData& v){
  GAGeneticAlgorithm::objectiveData(v);
  for(int i=0; i<pop->size(); i++)
    pop->individual(i).evalData(v);
}

const GAPopulation&
GADescriptorGA::population(const GAPopulation& p) {
  if(p.size() < 1) {
    GAErr(GA_LOC, className(), "population", gaErrNoIndividuals);
    return *pop;
  }

  GAGeneticAlgorithm::population(p);
  oldPop->copy(*pop->clone());
  oldPop->geneticAlgorithm(*this);

  return *pop;
}

int
GADescriptorGA::populationSize(unsigned int n) {
  GAGeneticAlgorithm::populationSize(n);
  oldPop->size(n);
  return n;
}

int
GADescriptorGA::minimaxi(int m) {
  GAGeneticAlgorithm::minimaxi(m);
  if(m == MINIMIZE)
    oldPop->order(GAPopulation::LOW_IS_BEST);
  else
    oldPop->order(GAPopulation::HIGH_IS_BEST);
  return minmax;
}





// Initialize the population, set the random seed as needed, do a few stupidity
// checks, reset the stats.  We must initialize the old pop because there is no
// guarantee that each individual will get initialized during the course of our
// operator++ operations.  We do not evaluate the old pop because that will
// happen as-needed later on.
void
GADescriptorGA::initialize(unsigned int seed)
{
  GARandomSeed(seed);

  pop->initialize();


#pragma omp parallel for num_threads(maxCoresUsed)
  for(int p_it=0;p_it<pop->size();++p_it){

  pop->individual(p_it).evaluate(gaTrue);
  }

//  pop->evaluate(gaTrue);	// the old pop will get it when the pops switch
//  oldPop->initialize();

  stats.reset(*pop);

  if(!scross)
    GAErr(GA_LOC, className(), "initialize", gaErrNoSexualMating);
}


//   Evolve a new generation of genomes.  When we start this routine, pop
// contains the current generation.  When we finish, pop contains the new
// generation and oldPop contains the (no longer) current generation.  The
// previous old generation is lost.  We don't deallocate any memory, we just
// reset the contents of the genomes.
//   The selection routine must return a pointer to a genome from the old
// population.
void
GADescriptorGA::step()
{
  int i, mut, c1, c2;  
  GAGenome *mom, *dad;          // tmp holders for selected genomes

  GAPopulation *tmppop;		// Swap the old population with the new pop.
  tmppop = oldPop;		// When we finish the ++ we want the newly
  oldPop = pop;			// generated population to be current (for
  pop = tmppop;			// references to it from member functions).

// Generate the individuals in the temporary population from individuals in
// the main population.

  for(i=0; i<pop->size()-1; i+=2){	// takes care of odd population
    mom = &(oldPop->select());
    dad = &(oldPop->select());
    stats.numsel += 2;		// keep track of number of selections

    c1 = c2 = 0;
    if(GAFlipCoin(pCrossover())){
      stats.numcro += (*scross)(*mom, *dad,
                &pop->individual(i), &pop->individual(i+1));
      c1 = c2 = 1;
    }
    else{
      pop->individual( i ).copy(*mom);
      pop->individual(i+1).copy(*dad);
    }
    stats.nummut += (mut = pop->individual( i ).mutate(pMutation()));
    if(mut > 0) c1 = 1;
    stats.nummut += (mut = pop->individual(i+1).mutate(pMutation()));
    if(mut > 0) c2 = 1;

    stats.numeval += c1 + c2;
  }
  if(pop->size() % 2 != 0){	// do the remaining population member
    mom = &(oldPop->select());
    dad = &(oldPop->select());
    stats.numsel += 2;		// keep track of number of selections

    c1 = 0;
    if(GAFlipCoin(pCrossover())){
      stats.numcro += (*scross)(*mom, *dad, &pop->individual(i), (GAGenome*)0);
      c1 = 1;
    }
    else{
      if(GARandomBit())
    pop->individual( i ).copy(*mom);
      else
    pop->individual( i ).copy(*dad);
    }
    stats.nummut += (mut = pop->individual( i ).mutate(pMutation()));
    if(mut > 0) c1 = 1;

    stats.numeval += c1;
  }

  stats.numrep += pop->size();

#pragma omp parallel for num_threads(maxCoresUsed)
  for(int p_it=0;p_it<pop->size();++p_it){

  pop->individual(p_it).evaluate(gaTrue);
  }

//  pop->evaluate(gaTrue);	// get info about current pop for next time

// If we are supposed to be elitist, carry the best individual from the old
// population into the current population.  Be sure to check whether we are
// supposed to minimize or maximize.

  if(minimaxi() == GAGeneticAlgorithm::MAXIMIZE) {
    if(el && oldPop->best().score() > pop->best().score())
      oldPop->replace(pop->replace(&(oldPop->best()), GAPopulation::WORST),
              GAPopulation::BEST);
  }
  else {
    if(el && oldPop->best().score() < pop->best().score())
      oldPop->replace(pop->replace(&(oldPop->best()), GAPopulation::WORST),
              GAPopulation::BEST);
  }

  stats.update(*pop);		// update the statistics by one generation
}
