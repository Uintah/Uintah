#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompMooneyRivlin.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHook.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookPlas.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoScram.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoScramForBinder.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MWViscoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Membrane.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HyperElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
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
   
   if (integrator != "implicit" && integrator != "explicit" 
		   	&& integrator != "fracture"){
     string txt="MPM: time integrator [explicit or implicit] hasn't been set.";
    throw ProblemSetupException(txt);
   }   
   
   if (mat_type == "comp_mooney_rivlin")
      return(scinew CompMooneyRivlin(child,lb,n8or27));
   
   else if (mat_type ==  "comp_neo_hook") {
     if (integrator == "explicit" || integrator == "fracture")
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
   
   else if (mat_type ==  "visco_scram_binder")
      return(scinew ViscoScramForBinder(child,lb,n8or27));
   
   else if (mat_type ==  "hypo_elastic") {
     if (integrator == "explicit" || integrator == "fracture")
      return(scinew HypoElastic(child,lb,n8or27));
     else if (integrator == "implicit")
       return(scinew HypoElasticImplicit(child,lb,n8or27));
   }
   
   else if (mat_type ==  "mw_visco_elastic")
      return(scinew MWViscoElastic(child,lb,n8or27));
   
   else if (mat_type ==  "membrane")
      return(scinew Membrane(child,lb,n8or27));
   
   else if (mat_type ==  "shell_CNH")
      return(scinew ShellMaterial(child,lb,n8or27));
   
   else if (mat_type ==  "hypoelastic_plastic")
      return(scinew HypoElasticPlastic(child,lb,n8or27));
   
   else if (mat_type ==  "hyperelastic_plastic")
      return(scinew HyperElasticPlastic(child,lb,n8or27));
   
   else 
      throw ProblemSetupException("Unknown Material Type R ("+mat_type+")");

   return 0;
}
