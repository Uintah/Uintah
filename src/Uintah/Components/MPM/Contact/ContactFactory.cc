#include "ContactFactory.h"
#include "NullContact.h"
#include "SingleVelContact.h"
#include "FrictionContact.h"
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <string>
using std::cerr;


using namespace Uintah::MPM;

Contact* ContactFactory::create(const ProblemSpecP& ps, SimulationStateP &ss)
{

   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

   for (ProblemSpecP child = mpm_ps->findBlock("contact"); child != 0;
	child = child->findNextBlock("contact")) {
      std::string con_type;
      child->require("type",con_type);
      cerr << "con_type = " << con_type << std::endl;
    
      if (con_type == "null")
	 return(scinew NullContact(child,ss));
      
      else if (con_type == "single_velocity")
	 return(scinew SingleVelContact(child,ss));

      else if (con_type == "friction")
	 return(scinew FrictionContact(child,ss));
    
      else {
	 cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
	 //      exit(1);
      }
    
   }
   return 0;
}

// $Log$
// Revision 1.6  2000/09/25 18:08:22  sparker
// include datawarehouse.h for template instantiation
//
// Revision 1.5  2000/07/11 19:18:23  tan
// Added #include <Uintah/Interface/ProblemSpec.h> into the .cc file.
// Some compiling patterns need it.
//
// Revision 1.4  2000/05/30 20:19:08  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.2  2000/04/28 21:08:25  jas
// Added exception to the creation of Contact factory if contact is not
// specified.
//
// Revision 1.1  2000/04/27 21:28:57  jas
// Contact is now created using a factory.
//
