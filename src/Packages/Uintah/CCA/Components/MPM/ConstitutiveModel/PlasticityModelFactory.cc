#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IsoHardeningPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MTSPlastic.h>
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
   else if (mat_type == "mts_model")
      return(scinew MTSPlastic(child));
   else 
      throw ProblemSetupException("Unknown Plasticity Model ("+mat_type+")");
}
