#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FirstOrderCEAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderAdvector.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/SecondOrderCEAdvector.h>
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

Advector* AdvectionFactory::create(ProblemSpecP& ps,
                                   std::string& advect_type)
{
    ProblemSpecP child = ps->findBlock("advection");
    if(!child)
      throw ProblemSetupException("Cannot find advection tag");

    if(!child->getAttribute("type",advect_type))
      throw ProblemSetupException("No type for advection"); 
    
    if (advect_type == "FirstOrder") 
      return(scinew FirstOrderAdvector());
    else if (advect_type == "FirstOrderCE") 
      return(scinew FirstOrderCEAdvector());
    else if (advect_type == "SecondOrder") 
      return(scinew SecondOrderAdvector());

    else if (advect_type == "SecondOrderCE") 
      return(scinew SecondOrderCEAdvector());
    else
      throw ProblemSetupException("Unknown advection Type R ("+advect_type+")");

}
