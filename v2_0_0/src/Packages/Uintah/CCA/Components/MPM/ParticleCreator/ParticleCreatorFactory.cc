#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCook.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
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

ConstitutiveModel* ConstitutiveModelFactory::create(ProblemSpecP& ps,
						    MPMLabel* lb, int n8or27,
						    string integrator)
{
   ProblemSpecP child = ps->findBlock("constitutive_model");
   if(!child)
      throw ProblemSetupException("Cannot find constitutive_model tag");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for constitutive_model");
   
   if (integrator != "implicit" && integrator != "explicit"){
     string txt="MPM: time integrator [explicit or implicit] hasn't been set.";
    throw ProblemSetupException(txt);
   }   
   
   if (mat_type == "comp_mooney_rivlin")
      return(scinew CompMooneyRivlin(child,lb,n8or27));
   
   else if (mat_type ==  "comp_neo_hook") {
     if (integrator == "explicit")
      return(scinew CompNeoHook(child,lb,n8or27));
     else if (integrator == "implicit") 
       return(scinew CompNeoHookImplicit(child,lb,n8or27));
   }
      
   else if (mat_type ==  "ideal_gas")
      return(scinew IdealGasMP(child,lb,n8or27));
      
   else if (mat_type == "comp_neo_hook_plastic")
      return(scinew CompNeoHookPlas(child,lb,n8or27));
   
   else if (mat_type ==  "visco_scram")
      return(scinew ViscoScram(child,lb,n8or27));
   
   else if (mat_type ==  "hypo_elastic")
      return(scinew HypoElastic(child,lb,n8or27));
   
   else if (mat_type ==  "mw_visco_elastic")
      return(scinew MWViscoElastic(child,lb,n8or27));
   
   else if (mat_type ==  "membrane")
      return(scinew Membrane(child,lb,n8or27));
   
   else if (mat_type ==  "johnson_cook")
      return(scinew JohnsonCook(child,lb,n8or27));
   
   else 
      throw ProblemSetupException("Unknown Material Type R ("+mat_type+")");

   return 0;
}
