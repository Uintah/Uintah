#include "ConstitutiveModelFactory.h"
#include "CompMooneyRivlin.h"
#include "CompNeoHook.h"
#include "CompNeoHookPlas.h"
#include "ViscoScram.h"
#include "HypoElastic.h"
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
							 MPMLabel* lb)
{
   ProblemSpecP child = ps->findBlock("constitutive_model");
   if(!child)
      throw ProblemSetupException("Cannot find constitutive_model tag");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for constitutive_model");
   
   if (mat_type == "comp_mooney_rivlin")
      return(scinew CompMooneyRivlin(child,lb));
   
   else if (mat_type ==  "comp_neo_hook")
      return(scinew CompNeoHook(child,lb));
      
   else if (mat_type == "comp_neo_hook_plastic")
      return(scinew CompNeoHookPlas(child,lb));
   
   else if (mat_type ==  "visco_scram")
      return(scinew ViscoScram(child,lb));
   
   else if (mat_type ==  "hypo_elastic")
      return(scinew HypoElastic(child,lb));
   
   else 
      throw ProblemSetupException("Unknown Material Type R ("+mat_type+")");
}
