#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/NullContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/SingleVelContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/FrictionContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/RigidBodyContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ApproachContact.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <string>
using std::cerr;


using namespace Uintah;

Contact* ContactFactory::create(const ProblemSpecP& ps, SimulationStateP &ss,
                                                    MPMLabel* lb, int n8or27)
{

   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

   for (ProblemSpecP child = mpm_ps->findBlock("contact"); child != 0;
	child = child->findNextBlock("contact")) {
      std::string con_type;
      child->require("type",con_type);
    
      if (con_type == "null")
	 return(scinew NullContact(child,ss,lb));
      
      else if (con_type == "single_velocity")
	 return(scinew SingleVelContact(child,ss,lb));

      else if (con_type == "friction")
	 return(scinew FrictionContact(child,ss,lb,n8or27));
    
      else if (con_type == "approach")
	 return(scinew ApproachContact(child,ss,lb,n8or27));

      else if (con_type == "rigid")
	 return(scinew RigidBodyContact(child,ss,lb));
    
      else {
	 cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
        throw ProblemSetupException(" E R R O R----->MPM:Unknown Contact type");
	 //      exit(1);
      }
    
   }
   return 0;
}

