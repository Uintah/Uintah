#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/NullContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/SingleVelContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/FrictionContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ApproachContact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/CompositeContact.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <string>
using std::cerr;


using namespace Uintah;

Contact* ContactFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, SimulationStateP &ss,
				MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");
   
   CompositeContact * contact_list = scinew CompositeContact(myworld, lb, flag);
   
   for (ProblemSpecP child = mpm_ps->findBlock("contact"); child != 0;
	child = child->findNextBlock("contact")) {
     
     std::string con_type;
     child->getWithDefault("type",con_type, "null");
     
      if (con_type == "null")
        contact_list->add(scinew NullContact(myworld,ss,lb,flag));
      
      else if (con_type == "single_velocity")
        contact_list->add(scinew SingleVelContact(myworld,child,ss,lb,flag));
      
      else if (con_type == "friction")
        contact_list->add(scinew FrictionContact(myworld,child,ss,lb,flag));
      
      else if (con_type == "approach")
        contact_list->add(scinew ApproachContact(myworld,child,ss,lb,flag));
      
      else if (con_type == "specified_velocity" || con_type == "specified"
                                                || con_type == "rigid"  )
        contact_list->add(scinew SpecifiedBodyContact(myworld,child,ss,lb,flag));
      
      else {
        cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
        throw ProblemSetupException(" E R R O R----->MPM:Unknown Contact type");
      }
   }
   
   // 
   if(contact_list->size()==0) {
     cout << "no contact - using null" << endl;
     contact_list->add(scinew NullContact(myworld,ss,lb,flag));
   }
   
   return contact_list;
}
