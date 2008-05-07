
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
  ProblemSpecP turb_ps = ps->findBlock("turbulence");
  
  if(turb_ps){
    std::string turbulence_model;
    if(!turb_ps->getAttribute("model",turbulence_model)){
      throw ProblemSetupException("No model for turbulence", __FILE__, __LINE__); 
    }
    if (turbulence_model == "Smagorinsky"){
      return(scinew Smagorinsky_Model(turb_ps, sharedState));
    }else if (turbulence_model == "Germano"){ 
      return(scinew DynamicModel(turb_ps, sharedState));
    }else{
      ostringstream warn;
      warn << "ERROR ICE: Unknown turbulence model ("<< turbulence_model << " )\n"
         << "Valid models are:\n" 
         << "Smagorinsky\n"
         << "Germano\n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  }
  return 0;
}
