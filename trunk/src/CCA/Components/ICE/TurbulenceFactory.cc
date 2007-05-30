
#include <CCA/Components/ICE/TurbulenceFactory.h>
#include <CCA/Components/ICE/SmagorinskyModel.h>
#include <CCA/Components/ICE/DynamicModel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <SCIRun/Core/Malloc/Allocator.h>

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
        throw ProblemSetupException("No model for turbulence", __FILE__, __LINE__); 
    
      if (turbulence_model == "Smagorinsky") 
        return(scinew Smagorinsky_Model(child, sharedState));
      else if (turbulence_model == "Germano") 
        return(scinew DynamicModel(child, sharedState));
      else
        throw ProblemSetupException("Unknown turbulence model ("+turbulence_model+")", __FILE__, __LINE__);
    }
    return 0;
}
