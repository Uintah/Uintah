#include "YieldConditionFactory.h"
#include "VonMisesYield.h"
#include "GursonYield.h"
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
   else 
      throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Yield Condition ("+mat_type+")");
}

YieldCondition* 
YieldConditionFactory::createCopy(const YieldCondition* yc)
{
   if (dynamic_cast<const VonMisesYield*>(yc))
      return(scinew VonMisesYield(dynamic_cast<const VonMisesYield*>(yc)));

   else if (dynamic_cast<const GursonYield*>(yc))
      return(scinew GursonYield(dynamic_cast<const GursonYield*>(yc)));

   else 
      throw ProblemSetupException("Cannot create copy of unknown yield condition");
}
