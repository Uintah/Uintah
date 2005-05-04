#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/RigidMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompMooneyRivlin.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHook.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CNHDamage.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CNHPDamage.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/TransIsoHyper.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/TransIsoHyperImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoTransIsoHyper.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookPlas.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoScram.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoSCRAMHotSpot.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MWViscoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Membrane.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
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
                                                    MPMLabel* lb, 
                                                    MPMFlags* flags)
{
  ProblemSpecP child = ps->findBlock("constitutive_model");
  if(!child)
    throw ProblemSetupException("Cannot find constitutive_model tag");
  string mat_type;
  if(!child->getAttribute("type", mat_type))
    throw ProblemSetupException("No type for constitutive_model");
   
  if (flags->d_integrator_type != "implicit" && 
      flags->d_integrator_type != "explicit" && 
      flags->d_integrator_type != "fracture"){
    string txt="MPM: time integrator [explicit or implicit] hasn't been set.";
    throw ProblemSetupException(txt);
  }   
   
  if (mat_type == "rigid")
    return(scinew RigidMaterial(child,lb,flags));

  else if (mat_type == "comp_mooney_rivlin")
    return(scinew CompMooneyRivlin(child,lb,flags));
   
  else if (mat_type ==  "comp_neo_hook") {
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew CompNeoHook(child,lb,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew CompNeoHookImplicit(child,lb,flags));
  }
  else if (mat_type ==  "cnh_damage") 
    return(scinew CNHDamage(child,lb,flags));

  else if (mat_type ==  "cnhp_damage") 
    return(scinew CNHPDamage(child,lb,flags));

  else if (mat_type ==  "trans_iso_hyper") {
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew TransIsoHyper(child,lb,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew TransIsoHyperImplicit(child,lb,flags));
  }
  
  else if (mat_type ==  "visco_trans_iso_hyper")
    return(scinew ViscoTransIsoHyper(child,lb,flags));

  else if (mat_type ==  "ideal_gas")
    return(scinew IdealGasMP(child,lb,flags));

  else if (mat_type == "comp_neo_hook_plastic")
    return(scinew CompNeoHookPlas(child,lb,flags));
   
  else if (mat_type ==  "visco_scram")
    return(scinew ViscoScram(child,lb,flags));
   
  else if (mat_type ==  "viscoSCRAM_hs")
    return(scinew ViscoSCRAMHotSpot(child,lb,flags));
   
  else if (mat_type ==  "hypo_elastic") {
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew HypoElastic(child,lb,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew HypoElasticImplicit(child,lb,flags));
  }

  else if (mat_type ==  "mw_visco_elastic")
    return(scinew MWViscoElastic(child,lb,flags));
   
  else if (mat_type ==  "membrane")
    return(scinew Membrane(child,lb,flags));

  else if (mat_type ==  "shell_CNH")
    return(scinew ShellMaterial(child,lb,flags));
   
  else if (mat_type ==  "hypoelastic_plastic")
    return(scinew HypoElasticPlastic(child,lb,flags));

  else if (mat_type ==  "elastic_plastic")
    return(scinew ElasticPlastic(child,lb,flags));

  else 
    throw ProblemSetupException("Unknown Material Type R ("+mat_type+")");

  return 0;
}

/* Create a copy of the relevant constitutive model */
ConstitutiveModel* 
ConstitutiveModelFactory::createCopy(const ConstitutiveModel* cm)
{
  if (dynamic_cast<const RigidMaterial*>(cm)) 
    return(scinew RigidMaterial(dynamic_cast<const RigidMaterial*>(cm)));
   
  else if (dynamic_cast<const CompMooneyRivlin*>(cm)) 
    return(scinew CompMooneyRivlin(dynamic_cast<const CompMooneyRivlin*>(cm)));
   
  else if (dynamic_cast<const TransIsoHyper*>(cm)) 
    return(scinew TransIsoHyper(dynamic_cast<const TransIsoHyper*>(cm)));

  else if (dynamic_cast<const TransIsoHyperImplicit*>(cm)) 
    return(scinew TransIsoHyperImplicit(dynamic_cast<const TransIsoHyperImplicit*>(cm)));

  else if (dynamic_cast<const CompNeoHook*>(cm)) 
    return(scinew CompNeoHook(dynamic_cast<const CompNeoHook*>(cm)));

  else if (dynamic_cast<const CNHDamage*>(cm)) 
    return(scinew CNHDamage(dynamic_cast<const CNHDamage*>(cm)));

  else if (dynamic_cast<const CNHPDamage*>(cm)) 
    return(scinew CNHPDamage(dynamic_cast<const CNHPDamage*>(cm)));

  else if (dynamic_cast<const CompNeoHookImplicit*>(cm)) 
    return(scinew CompNeoHookImplicit(dynamic_cast<const CompNeoHookImplicit*>(cm)));

  else if (dynamic_cast<const IdealGasMP*>(cm)) 
    return(scinew IdealGasMP(dynamic_cast<const IdealGasMP*>(cm)));

  else if (dynamic_cast<const CompNeoHookPlas*>(cm)) 
    return(scinew CompNeoHookPlas(dynamic_cast<const CompNeoHookPlas*>(cm)));
   
  else if (dynamic_cast<const ViscoScram*>(cm)) 
    return(scinew ViscoScram(dynamic_cast<const ViscoScram*>(cm)));
   
  else if (dynamic_cast<const ViscoSCRAMHotSpot*>(cm)) 
    return(scinew ViscoSCRAMHotSpot(dynamic_cast<const ViscoSCRAMHotSpot*>(cm)));
   
  else if (dynamic_cast<const HypoElastic*>(cm)) 
    return(scinew HypoElastic(dynamic_cast<const HypoElastic*>(cm)));
   
  else if (dynamic_cast<const HypoElasticImplicit*>(cm)) 
    return(scinew HypoElasticImplicit(dynamic_cast<const HypoElasticImplicit*>(cm)));
   
  else if (dynamic_cast<const MWViscoElastic*>(cm)) 
    return(scinew MWViscoElastic(dynamic_cast<const MWViscoElastic*>(cm)));
   
  else if (dynamic_cast<const Membrane*>(cm)) 
    return(scinew Membrane(dynamic_cast<const Membrane*>(cm)));

  else if (dynamic_cast<const ShellMaterial*>(cm)) 
    return(scinew ShellMaterial(dynamic_cast<const ShellMaterial*>(cm)));
   
  else if (dynamic_cast<const HypoElasticPlastic*>(cm)) 
    return(scinew HypoElasticPlastic(dynamic_cast<const HypoElasticPlastic*>(cm)));
   
  else if (dynamic_cast<const ElasticPlastic*>(cm)) 
    return(scinew ElasticPlastic(dynamic_cast<const ElasticPlastic*>(cm)));

  else 
    throw ProblemSetupException("Cannot create copy of unknown material.");

  return 0;
}
