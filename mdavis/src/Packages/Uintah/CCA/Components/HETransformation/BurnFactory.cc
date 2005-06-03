#include <Packages/Uintah/CCA/Components/HETransformation/BurnFactory.h>
#include <Packages/Uintah/CCA/Components/HETransformation/NullBurn.h>
#include <Packages/Uintah/CCA/Components/HETransformation/SimpleBurn.h>
#include <Packages/Uintah/CCA/Components/HETransformation/PressureBurn.h>
#include <Packages/Uintah/CCA/Components/HETransformation/IgnitionCombustion.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

Burn* BurnFactory::create(ProblemSpecP& ps)
{
    ProblemSpecP child = ps->findBlock("burn");
    if(!child)
      throw ProblemSetupException("Cannot find burn_model tag");
    std::string burn_type;
    if(!child->getAttribute("type",burn_type))
      throw ProblemSetupException("No type for burn_model"); 
    
    if (burn_type == "null")
      return(scinew NullBurn(child));
    
    else if (burn_type == "pressure")
      return(scinew PressureBurn(child));

    else if (burn_type == "simple")
      return(scinew SimpleBurn(child));

    else if (burn_type == "IgnitionCombustion")
      return(scinew IgnitionCombustion(child));
    
    else 
      throw ProblemSetupException("Unknown Burn Type R ("+burn_type+")");
}

