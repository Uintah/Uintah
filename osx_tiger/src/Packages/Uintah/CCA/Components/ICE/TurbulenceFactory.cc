
#include <Packages/Uintah/CCA/Components/ICE/TurbulenceFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/ICE/DynamicModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

TurbulenceFactory::TurbulenceFactory()
{
}

TurbulenceFactory::~TurbulenceFactory()
{
}

Turbulence* TurbulenceFactory::create(ProblemSpecP& ps, SimulationStateP& sharedState)
{
    ProblemSpecP child = ps->findBlock("turbulence");
   
    if(child){
      std::string turbulence_model;
      if(!child->getAttribute("model",turbulence_model))
        throw ProblemSetupException("No model for turbulence"); 
    
      if (turbulence_model == "Smagorinsky") 
        return(scinew Smagorinsky_Model(child, sharedState));
      else if (turbulence_model == "Germano") 
        return(scinew DynamicModel(child, sharedState));
      else
        throw ProblemSetupException("Unknown turbulence model ("+turbulence_model+")");
    }
    return 0;
}
