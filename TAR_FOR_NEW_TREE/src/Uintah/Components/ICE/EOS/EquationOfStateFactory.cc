#include "EquationOfStateFactory.h"
#include "IdealGas.h"
#include "Harlow.h"
#include "StiffGas.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah::ICESpace;

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
	 cerr << "Unknown Material Type R (" << mat_type << ")" << std::endl;
	 //      exit(1);
      }
   }
   return 0;
}
//$Log$
//Revision 1.4  2000/10/31 04:27:42  jas
//Fix typo.
//
//Revision 1.3  2000/10/31 04:14:28  jas
//Added stiff gas EOS type.  It is just a copy of IdealGas.
//
//Revision 1.2  2000/10/26 23:43:14  jas
//Added Harlow to factory.
//
//Revision 1.1  2000/10/06 04:02:16  jas
//Move into a separate EOS directory.
//
//Revision 1.3  2000/10/04 23:42:29  jas
//Add IdealGas to the EOS factory.
//
//Revision 1.2  2000/10/04 20:17:52  jas
//Change namespace ICE to ICESpace.
//
//Revision 1.1  2000/10/04 19:26:14  guilkey
//Initial commit of some classes to help mainline ICE.
//




