#include "PlasticityModelFactory.h"                                             
#include "IsoHardeningPlastic.h"
#include "JohnsonCookPlastic.h"
#include "ZerilliArmstrongPlastic.h"
#include "MTSPlastic.h"
#include "SCGPlastic.h"
#include "PTWPlastic.h"
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

PlasticityModel* PlasticityModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("plasticity_model");
   if(!child)
      throw ProblemSetupException("Cannot find plasticity_model tag");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for plasticity_model");
   if (mat_type == "isotropic_hardening")
      return(scinew IsoHardeningPlastic(child));
   else if (mat_type == "johnson_cook")
      return(scinew JohnsonCookPlastic(child));
   else if (mat_type == "zerilli_armstrong")
      return(scinew ZerilliArmstrongPlastic(child));
   else if (mat_type == "mts_model")
      return(scinew MTSPlastic(child));
   else if (mat_type == "steinberg_cochran_guinan")
      return(scinew SCGPlastic(child));
   else if (mat_type == "preston_tonks_wallace")
      return(scinew PTWPlastic(child));
   else 
      throw ProblemSetupException("Unknown Plasticity Model ("+mat_type+")");
}

PlasticityModel* 
PlasticityModelFactory::createCopy(const PlasticityModel* pm)
{
   if (dynamic_cast<const IsoHardeningPlastic*>(pm))
      return(scinew IsoHardeningPlastic(dynamic_cast<const 
					IsoHardeningPlastic*>(pm)));

   else if (dynamic_cast<const JohnsonCookPlastic*>(pm))
      return(scinew JohnsonCookPlastic(dynamic_cast<const 
				       JohnsonCookPlastic*>(pm)));

   else if (dynamic_cast<const ZerilliArmstrongPlastic*>(pm))
      return(scinew ZerilliArmstrongPlastic(dynamic_cast<const 
					    ZerilliArmstrongPlastic*>(pm)));

   else if (dynamic_cast<const MTSPlastic*>(pm))
      return(scinew MTSPlastic(dynamic_cast<const MTSPlastic*>(pm)));
   
   else if (dynamic_cast<const SCGPlastic*>(pm))
      return(scinew SCGPlastic(dynamic_cast<const SCGPlastic*>(pm)));

   else if (dynamic_cast<const PTWPlastic*>(pm))
      return(scinew PTWPlastic(dynamic_cast<const PTWPlastic*>(pm)));
   
   else 
      throw ProblemSetupException("Cannot create copy of unknown plasticity model");
}

