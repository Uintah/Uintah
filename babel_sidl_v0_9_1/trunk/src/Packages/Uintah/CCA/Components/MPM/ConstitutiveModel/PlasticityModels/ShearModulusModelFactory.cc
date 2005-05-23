#include "ShearModulusModelFactory.h"
#include "ConstantShear.h"
#include "MTSShear.h"
#include "SCGShear.h"
#include "PTWShear.h"
#include "NPShear.h"
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
ShearModulusModel* ShearModulusModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("shear_modulus_model");
   if(!child)
      throw ProblemSetupException("MPM::ConstitutiveModel:Cannot find shear modulus model.");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("MPM::ConstitutiveModel:No type for shear modulus model.");
   
   if (mat_type == "constant_shear")
      return(scinew ConstantShear(child));
   else if (mat_type == "mts_shear")
      return(scinew MTSShear(child));
   else if (mat_type == "scg_shear")
      return(scinew SCGShear(child));
   else if (mat_type == "ptw_shear")
      return(scinew PTWShear(child));
   else if (mat_type == "np_shear")
      return(scinew NPShear(child));
   else 
      throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Shear Modulus Model ("+mat_type+")");
}

ShearModulusModel* 
ShearModulusModelFactory::createCopy(const ShearModulusModel* smm)
{
   if (dynamic_cast<const ConstantShear*>(smm))
      return(scinew ConstantShear(dynamic_cast<const ConstantShear*>(smm)));
   else if (dynamic_cast<const MTSShear*>(smm))
      return(scinew MTSShear(dynamic_cast<const MTSShear*>(smm)));
   else if (dynamic_cast<const SCGShear*>(smm))
      return(scinew SCGShear(dynamic_cast<const SCGShear*>(smm)));
   else if (dynamic_cast<const PTWShear*>(smm))
      return(scinew PTWShear(dynamic_cast<const PTWShear*>(smm)));
   else if (dynamic_cast<const NPShear*>(smm))
      return(scinew NPShear(dynamic_cast<const NPShear*>(smm)));
   else 
      throw ProblemSetupException("Cannot create copy of unknown shear modulus model");
}
