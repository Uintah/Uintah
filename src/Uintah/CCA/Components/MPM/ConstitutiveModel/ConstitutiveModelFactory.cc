/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoTransIsoHyperImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookPlas.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoScram.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoScramImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoSCRAMHotSpot.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticFortran.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Kayenta.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticImplicit.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MWViscoElastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Membrane.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ElasticPlasticHP.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/SmallStrainPlastic.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/SoilFoam.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Water.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ViscoPlastic.h>
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
                                                    MPMFlags* flags)
{
  ProblemSpecP child = ps->findBlock("constitutive_model");
  if(!child)
    throw ProblemSetupException("Cannot find constitutive_model tag", __FILE__, __LINE__);
  string mat_type;
  if(!child->getAttribute("type", mat_type))
    throw ProblemSetupException("No type for constitutive_model", __FILE__, __LINE__);
   
  if (flags->d_integrator_type != "implicit" && 
      flags->d_integrator_type != "explicit" && 
      flags->d_integrator_type != "fracture"){
    string txt="MPM: time integrator [explicit or implicit] hasn't been set.";
    throw ProblemSetupException(txt, __FILE__, __LINE__);
  }   

  if (mat_type == "rigid")
    return(scinew RigidMaterial(child,flags));

  else if (mat_type == "comp_mooney_rivlin")
    return(scinew CompMooneyRivlin(child,flags));
   
  else if (mat_type ==  "comp_neo_hook") {
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew CompNeoHook(child,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew CompNeoHookImplicit(child,flags));
  }
  else if (mat_type ==  "cnh_damage") 
    return(scinew CNHDamage(child,flags));

  else if (mat_type ==  "cnhp_damage") 
    return(scinew CNHPDamage(child,flags));

  else if (mat_type ==  "trans_iso_hyper") {
    if (flags->d_integrator_type == "explicit" ||
        flags->d_integrator_type == "fracture")
      return(scinew TransIsoHyper(child,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew TransIsoHyperImplicit(child,flags));
  }
  
  else if (mat_type ==  "visco_trans_iso_hyper") {
    if (flags->d_integrator_type == "explicit" ||
        flags->d_integrator_type == "fracture")
      return(scinew ViscoTransIsoHyper(child,flags));
    else if (flags->d_integrator_type == "implicit")
    return(scinew ViscoTransIsoHyperImplicit(child,flags));
  }
  
  else if (mat_type ==  "ideal_gas")
    return(scinew IdealGasMP(child,flags));

  else if (mat_type ==  "water")
    return(scinew Water(child,flags));

  else if (mat_type == "comp_neo_hook_plastic")
    return(scinew CompNeoHookPlas(child,flags));
   
  else if (mat_type ==  "visco_scram"){
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew ViscoScram(child,flags));
    else if (flags->d_integrator_type == "implicit")
      return(scinew ViscoScramImplicit(child,flags));
  }
   
  else if (mat_type ==  "viscoSCRAM_hs")
    return(scinew ViscoSCRAMHotSpot(child,flags));
   
  else if (mat_type ==  "hypo_elastic") {
    if (flags->d_integrator_type == "explicit" || 
        flags->d_integrator_type == "fracture")
      return(scinew HypoElastic(child,flags));
    else if (flags->d_integrator_type == "implicit"){
      if(!flags->d_doGridReset){
         ostringstream msg;
         msg << "\n ERROR: One may not use HypoElastic along with \n"
             << " <do_grid_reset>false</do_grid_reset> \n";
         throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
      return(scinew HypoElasticImplicit(child,flags));
    }
  }

  else if (mat_type == "hypo_elastic_fortran")
    return(scinew HypoElasticFortran(child,flags));
   
  else if (mat_type == "kayenta")
    return(scinew Kayenta(child,flags));
   
  else if (mat_type ==  "mw_visco_elastic")
    return(scinew MWViscoElastic(child,flags));
   
  else if (mat_type ==  "membrane")
    return(scinew Membrane(child,flags));

  else if (mat_type ==  "shell_CNH")
    return(scinew ShellMaterial(child,flags));
   
  else if (mat_type ==  "hypoelastic_plastic")
    return(scinew HypoElasticPlastic(child,flags));

  else if (mat_type ==  "elastic_plastic")
    return(scinew ElasticPlastic(child,flags));

  else if (mat_type ==  "elastic_plastic_hp")
    return(scinew ElasticPlasticHP(child,flags));

  else if (mat_type ==  "small_strain_plastic")
    return(scinew SmallStrainPlastic(child,flags));

  else if (mat_type ==  "soil_foam")
    return(scinew SoilFoam(child,flags));

  else if (mat_type ==  "visco_plastic")
    return(scinew ViscoPlastic(child,flags));
  
  else 
    throw ProblemSetupException("Unknown Material Type R ("+mat_type+")", __FILE__, __LINE__);

  return 0;
}
