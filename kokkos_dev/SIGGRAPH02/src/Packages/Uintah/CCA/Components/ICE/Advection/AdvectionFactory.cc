#include "AdvectionFactory.h"
#include "FirstOrderAdvector.h"
#include "FirstOrderCEAdvector.h"
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

Advector* AdvectionFactory::create(ProblemSpecP& ps)
{
    ProblemSpecP child = ps->findBlock("advection");
    if(!child)
      throw ProblemSetupException("Cannot find advection tag");
    std::string advect_type;
    if(!child->getAttribute("type",advect_type))
      throw ProblemSetupException("No type for advection"); 
    
    if (advect_type == "FirstOrder") 
      return(scinew FirstOrderAdvector());
    else if (advect_type == "FirstOrderCE") 
      return(scinew FirstOrderCEAdvector());
    else
      throw ProblemSetupException("Unknown advection Type R ("+advect_type+")");

}
