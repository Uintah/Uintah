#include "ContactFactory.h"
#include "NullContact.h"
#include "SingleVelContact.h"
#include "FrictionContact.h"
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
	 return(new NullContact(child,ss));
      
      else if (con_type == "single_velocity")
	 return(new SingleVelContact(child,ss));

      else if (con_type == "friction")
	 return(new FrictionContact(child,ss));
    
      else {
	 cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
	 //      exit(1);
      }
    
   }
   return 0;
}

// $Log$
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
