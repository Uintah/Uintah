#include "EquationOfStateFactory.h"
#include "IdealGas.h"
#include "Harlow.h"
#include "StiffGas.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps)
{
   for (ProblemSpecP child = ps->findBlock(); child != 0;
	child = child->findNextBlock()) {
      std::string mat_type = child->getNodeName();
      if (mat_type == "ideal_gas") {
		 return(scinew IdealGas(child));
      }
      if (mat_type == "harlow") {
		 return(scinew Harlow(child));
      }
      if (mat_type == "stiff_gas") {
		 return(scinew StiffGas(child));
      } else {
	 cerr << "Unknown EOS Type R (" << mat_type << ")" << std::endl;
	 //      exit(1);
      }
   }
   return 0;
}
