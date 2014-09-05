#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/StabilityCheckFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/AcousticTensorCheck.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DruckerCheck.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/BeckerCheck.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DruckerBeckerCheck.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
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
      throw ProblemSetupException("Cannot find stability check criterion.");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for stability check criterion.");
   
   if (mat_type == "acoustic")
      return(scinew AcousticTensorCheck(child));
   else if (mat_type == "drucker")
      return(scinew DruckerCheck(child));
   else if (mat_type == "becker")
      return(scinew BeckerCheck(child));
   else if (mat_type == "drucker_becker")
      return(scinew DruckerBeckerCheck(child));
   else if (mat_type == "none")
      return 0;
   else 
      throw ProblemSetupException("Unknown Stability Check ("+mat_type+")");
}
