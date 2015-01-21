#include "MeltingTempModelFactory.h"
#include "ConstantMeltTemp.h"
#include "SCGMeltTemp.h"
#include "BPSMeltTemp.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace Uintah;

/// Create an instance of a Melting Temperature Model
MeltingTempModel* MeltingTempModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("melting_temp_model");
   if(!child)
      throw ProblemSetupException("MPM::ConstitutiveModel:Cannot find melting temp model.", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("MPM::ConstitutiveModel:No type for melting temp model.", __FILE__, __LINE__);
   
   if (mat_type == "constant_Tm")
      return(scinew ConstantMeltTemp(child));
   else if (mat_type == "scg_Tm")
      return(scinew SCGMeltTemp(child));
   else if (mat_type == "bps_Tm")
      return(scinew BPSMeltTemp(child));
   else 
      throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Melting Temp Model ("+mat_type+")",
                                  __FILE__, __LINE__);
}

MeltingTempModel* 
MeltingTempModelFactory::createCopy(const MeltingTempModel* mtm)
{
   if (dynamic_cast<const ConstantMeltTemp*>(mtm))
      return(scinew ConstantMeltTemp(dynamic_cast<const ConstantMeltTemp*>(mtm)));
   else if (dynamic_cast<const SCGMeltTemp*>(mtm))
      return(scinew SCGMeltTemp(dynamic_cast<const SCGMeltTemp*>(mtm)));
   else if (dynamic_cast<const BPSMeltTemp*>(mtm))
      return(scinew BPSMeltTemp(dynamic_cast<const BPSMeltTemp*>(mtm)));
   else 
      throw ProblemSetupException("Cannot create copy of unknown melting temp model", __FILE__, __LINE__);
}
