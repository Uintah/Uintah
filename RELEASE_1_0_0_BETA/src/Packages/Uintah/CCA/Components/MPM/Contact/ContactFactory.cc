#include "ContactFactory.h"
#include "NullContact.h"
#include "SingleVelContact.h"
#include "FrictionContact.h"
#include "RigidBodyContact.h"
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <string>
using std::cerr;


using namespace Uintah;

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
    
      else if (con_type == "rigid")
	 return(scinew RigidBodyContact(child,ss));
    
      else {
	 cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
	 //      exit(1);
      }
    
   }
   return 0;
}

