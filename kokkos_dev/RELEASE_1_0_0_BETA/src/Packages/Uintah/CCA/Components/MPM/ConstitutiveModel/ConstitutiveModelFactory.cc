#include "ConstitutiveModelFactory.h"
#include "CompMooneyRivlin.h"
#include "CompNeoHook.h"
#include "CompNeoHookPlas.h"
#include "ViscoScram.h"
#include "HypoElastic.h"
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

ConstitutiveModel* ConstitutiveModelFactory::create(ProblemSpecP& ps)
{
   for (ProblemSpecP child = ps->findBlock(); child != 0;
	child = child->findNextBlock()) {
      std::string mat_type = child->getNodeName();
      if (mat_type == "comp_mooney_rivlin")
	 return(scinew CompMooneyRivlin(child));
      
      else if (mat_type ==  "comp_neo_hook")
	 return(scinew CompNeoHook(child));
      
      else if (mat_type == "comp_neo_hook_plastic")
	 return(scinew CompNeoHookPlas(child));

      else if (mat_type ==  "visco_scram")
	 return(scinew ViscoScram(child));
      
      else if (mat_type ==  "hypo_elastic")
	 return(scinew HypoElastic(child));
      
      else {
	 cerr << "Unknown Material Type R (" << mat_type << ")" << std::endl;;
	 //      exit(1);
      }
   }
   return 0;
}
