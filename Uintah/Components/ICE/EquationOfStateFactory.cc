#include "EquationOfStateFactory.h"
#include <SCICore/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah::ICE;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps)
{
   for (ProblemSpecP child = ps->findBlock(); child != 0;
	child = child->findNextBlock()) {
      std::string mat_type = child->getNodeName();
      if (mat_type == "ideal_gas")
//	 return(scinew IdealGas(child));
      else {
	 cerr << "Unknown Material Type R (" << mat_type << ")" << std::endl;;
	 //      exit(1);
      }
   }
   return 0;
}
//$Log$
//Revision 1.1  2000/10/04 19:26:14  guilkey
//Initial commit of some classes to help mainline ICE.
//
