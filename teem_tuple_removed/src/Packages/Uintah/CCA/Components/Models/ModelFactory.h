
#ifndef Packages_Uintah_CCA_Components_Examples_ModelFactory_h
#define Packages_Uintah_CCA_Components_Examples_ModelFactory_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>

namespace Uintah {

/**************************************

CLASS
   ModelFactory
   
   ModelFactory simulation

GENERAL INFORMATION

   ModelFactory.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ModelFactory

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ModelFactory : public UintahParallelComponent, public ModelMaker {
  public:
    ModelFactory(const ProcessorGroup* myworld);
    virtual ~ModelFactory();

    virtual void makeModels(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP& sharedState,
			    std::vector<ModelInterface*>&);

  private:
    ModelFactory(const ModelFactory&);
    ModelFactory& operator=(const ModelFactory&);
	 
  };
}

#endif
