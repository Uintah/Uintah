
#include "MPMEquationOfStateFactory.h"
#include "DefaultHypoElasticEOS.h"
#include "MieGruneisenEOS.h"
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

MPMEquationOfState* MPMEquationOfStateFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("equation_of_state");
   if(!child)
      throw ProblemSetupException("Cannot find equation_of_state tag");
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for equation_of_state");
   
   if (mat_type == "mie_gruneisen")
      return(scinew MieGruneisenEOS(child));
   else if (mat_type == "default_hypo")
      return(scinew DefaultHypoElasticEOS(child));
   else 
      throw ProblemSetupException("Unknown MPMEquation of State Model ("+mat_type+")");

   //return 0;
}

MPMEquationOfState* 
MPMEquationOfStateFactory::createCopy(const MPMEquationOfState* eos)
{
   if (dynamic_cast<const MieGruneisenEOS*>(eos))
      return(scinew MieGruneisenEOS(dynamic_cast<const MieGruneisenEOS*>(eos)));

   else if (dynamic_cast<const DefaultHypoElasticEOS*>(eos))
      return(scinew DefaultHypoElasticEOS(dynamic_cast<const DefaultHypoElasticEOS*>(eos)));

   else 
      throw ProblemSetupException("Cannot create copy of unknown MPM EOS");

   //return 0;
}
