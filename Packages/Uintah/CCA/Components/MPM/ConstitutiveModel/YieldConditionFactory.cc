#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/YieldConditionFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/VonMisesYield.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/GursonYield.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/RousselierYield.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace std;
using namespace Uintah;

/// Create an instance of a Yield Condition.
/*! Available yield conditions are : von Mises, Gurson-Tvergaard-Needleman,
    Rosselier */
YieldCondition* YieldConditionFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("yield_condition");
   if(!child)
      throw ProblemSetupException("Cannot find yield condition.");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for yield condition.");
   
   if (mat_type == "vonMises")
      return(scinew VonMisesYield(child));
   else if (mat_type == "gurson")
      return(scinew GursonYield(child));
   else if (mat_type == "rousselier")
      return(scinew RousselierYield(child));
   else 
      throw ProblemSetupException("Unknown Yield Condition ("+mat_type+")");
}
