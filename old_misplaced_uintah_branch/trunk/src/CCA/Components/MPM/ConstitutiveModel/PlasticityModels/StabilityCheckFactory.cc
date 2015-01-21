#include "StabilityCheckFactory.h"
#include "DruckerCheck.h"
#include "BeckerCheck.h"
#include "NoneCheck.h"
#include "DruckerBeckerCheck.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;

/// Create an instance of a stabilty check method
/*! Available checks are : loss of ellipticity of the acoustic tensor */
StabilityCheck* StabilityCheckFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("stability_check");
   if(!child)
      throw ProblemSetupException("Cannot find stability check criterion.", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for stability check criterion.", __FILE__, __LINE__);
   
   if (mat_type == "drucker")
      return(scinew DruckerCheck(child));
   else if (mat_type == "becker")
      return(scinew BeckerCheck(child));
   else if (mat_type == "drucker_becker")
      return(scinew DruckerBeckerCheck(child));
   else if (mat_type == "none")
    return(scinew NoneCheck(child));
   else 
      throw ProblemSetupException("Unknown Stability Check ("+mat_type+")", __FILE__, __LINE__);
}

StabilityCheck* 
StabilityCheckFactory::createCopy(const StabilityCheck* sc)
{
   if (dynamic_cast<const DruckerCheck*>(sc))
      return(scinew DruckerCheck(dynamic_cast<const DruckerCheck*>(sc)));

   else if (dynamic_cast<const BeckerCheck*>(sc))
      return(scinew BeckerCheck(dynamic_cast<const BeckerCheck*>(sc)));

   else if (dynamic_cast<const DruckerBeckerCheck*>(sc))
      return(scinew DruckerBeckerCheck(dynamic_cast<const DruckerBeckerCheck*>(sc)));
   else if (dynamic_cast<const NoneCheck*>(sc))
      return(scinew NoneCheck(dynamic_cast<const NoneCheck*>(sc)));

   else 
      return 0;
}
