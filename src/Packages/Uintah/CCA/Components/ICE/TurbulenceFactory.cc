#include <Packages/Uintah/CCA/Components/ICE/TurbulenceFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/ICE/DynamicModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

Turbulence* TurbulenceFactory::create(ProblemSpecP& ps,
                                      bool& d_Turb)
{
    ProblemSpecP child = ps->findBlock("turbulence");
   
    if(child){
      d_Turb = true;
      std::string turbulence_model;
      if(!child->getAttribute("model",turbulence_model))
        throw ProblemSetupException("No model for turbulence"); 
    
      if (turbulence_model == "Smagorinsky") 
        return(scinew SmagorinskyModel(child));    
      else if (turbulence_model == "Germano") 
        return(scinew DynamicModel(child));   
      else
        throw ProblemSetupException("Unknown turbulence model ("+turbulence_model+")");
    }
    return 0;
}
