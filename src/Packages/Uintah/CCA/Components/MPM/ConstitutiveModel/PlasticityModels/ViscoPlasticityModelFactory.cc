#include "ViscoPlasticityModelFactory.h"                                        
#include "SuvicI.h"
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

ViscoPlasticityModel* ViscoPlasticityModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("visco_plasticity_model");
   if(!child)
      throw ProblemSetupException("Cannot find visco plasticity_model tag", __FILE__, __LINE__);
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for visco_plasticity_model", __FILE__, __LINE__);

if (mat_type == "suvic_i")
      return(scinew SuvicI(child));
   else 
      throw ProblemSetupException("Unknown ViscoPlasticity Model ("+mat_type+")", __FILE__, __LINE__);
}

ViscoPlasticityModel* 
ViscoPlasticityModelFactory::createCopy(const ViscoPlasticityModel* pm)
{

   
   if (dynamic_cast<const SuvicI*>(pm))
      return(scinew SuvicI(dynamic_cast<const SuvicI*>(pm)));
   
   else 
      throw ProblemSetupException("Cannot create copy of unknown Viscoplasticity model", __FILE__, __LINE__);
}

