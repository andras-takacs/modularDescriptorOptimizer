#ifndef GADESCRIPTORGA_H
#define GADESCRIPTORGA_H

#include <ga/GABaseGA.h>
#include <ga/gaid.h>


class GADescriptorGA: public GAGeneticAlgorithm {
    public:
      GADefineIdentity("GADescriptorGA", GAID::DescriptorGA)

      static GAParameterList& registerDefaultParameters(GAParameterList&);

    public:
      GADescriptorGA(const GAGenome&, const int&_maxCoresUsed);
      GADescriptorGA(const GAPopulation&);
      GADescriptorGA(const GADescriptorGA&);
      GADescriptorGA& operator=(const GADescriptorGA&);
      virtual ~GADescriptorGA();
      virtual void copy(const GAGeneticAlgorithm&);

      virtual void initialize(unsigned int seed=0);
      virtual void step();
      GADescriptorGA & operator++() { step(); return *this; }

      virtual int setptr(const char* name, const void* value);
      virtual int get(const char* name, void* value) const;

      GABoolean elitist() const {return el;}
      GABoolean elitist(GABoolean flag)
        {params.set(gaNelitism, (int)flag); return el=flag;}

      virtual int minimaxi() const {return minmax;}
      virtual int minimaxi(int m);

      virtual const GAPopulation& population() const {return *pop;}
      virtual const GAPopulation& population(const GAPopulation&);
      virtual int populationSize() const {return pop->size();}
      virtual int populationSize(unsigned int n);
      virtual GAScalingScheme& scaling() const {return pop->scaling();}
      virtual GAScalingScheme& scaling(const GAScalingScheme & s)
        {oldPop->scaling(s); return GAGeneticAlgorithm::scaling(s);}
      virtual GASelectionScheme& selector() const {return pop->selector(); }
      virtual GASelectionScheme& selector(const GASelectionScheme& s)
        {oldPop->selector(s); return GAGeneticAlgorithm::selector(s);}
      virtual void objectiveFunction(GAGenome::Evaluator f);
      virtual void objectiveData(const GAEvalData& v);

    protected:
      GAPopulation *oldPop;		// current and old populations
      GABoolean el;			// are we elitist?
      int maxCoresUsed;
    };



    #ifdef GALIB_USE_STREAMS
    inline STD_OSTREAM & operator<< (STD_OSTREAM & os, GADescriptorGA & arg)
    { arg.write(os); return(os); }
    inline STD_ISTREAM & operator>> (STD_ISTREAM & is, GADescriptorGA & arg)
    { arg.read(is); return(is); }
    #endif

    #endif // GADESCRIPTORGA_H
