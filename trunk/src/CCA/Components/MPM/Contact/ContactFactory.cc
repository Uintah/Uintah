#include <CCA/Components/MPM/Contact/ContactFactory.h>
#include <CCA/Components/MPM/Contact/NullContact.h>
#include <CCA/Components/MPM/Contact/SingleVelContact.h>
#include <CCA/Components/MPM/Contact/FrictionContact.h>
#include <CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
#include <CCA/Components/MPM/Contact/ApproachContact.h>
#include <CCA/Components/MPM/Contact/CompositeContact.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <SCIRun/Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>
using std::cerr;


using namespace Uintah;

Contact* ContactFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, SimulationStateP &ss,
				MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");
   
   
   if(!mpm_ps){
    string warn = "ERROR: Missing either <MaterialProperties> or <MPM> block from input file";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
   }
   
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
        contact_list->add(scinew SpecifiedBodyContact(myworld,child,ss,lb,
                                                      flag));
      
      else {
        cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
        throw ProblemSetupException(" E R R O R----->MPM:Unknown Contact type", __FILE__, __LINE__);
      }
   }
   
   // 
   if(contact_list->size()==0) {
     cout << "no contact - using null" << endl;
     contact_list->add(scinew NullContact(myworld,ss,lb,flag));
   }
   
   return contact_list;
}
