
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookDamage.h>
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

DamageModel* DamageModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("damage_model");
   if(!child)
      throw ProblemSetupException("Cannot find damage_model tag");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for damage_model");
   
   if (mat_type == "johnson_cook")
      return(scinew JohnsonCookDamage(child));
   else 
      throw ProblemSetupException("Unknown Damage Model ("+mat_type+")");

   //return 0;
}
