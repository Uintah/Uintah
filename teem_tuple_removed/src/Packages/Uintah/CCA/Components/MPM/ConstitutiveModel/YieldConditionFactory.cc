#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/YieldConditionFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/VonMisesYield.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/GursonYield.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/RousselierYield.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace Uintah;

/// Create an instance of a Yield Condition.
/*! Available yield conditions are : von Mises, Gurson-Tvergaard-Needleman,
    Rosselier */
YieldCondition* YieldConditionFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("yield_condition");
   if(!child)
      throw ProblemSetupException("MPM::ConstitutiveModel:Cannot find yield condition.");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("MPM::ConstitutiveModel:No type for yield condition.");
   
   if (mat_type == "vonMises")
      return(scinew VonMisesYield(child));
   else if (mat_type == "gurson")
      return(scinew GursonYield(child));
   else if (mat_type == "rousselier")
      return(scinew RousselierYield(child));
   else 
      throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Yield Condition ("+mat_type+")");
}
