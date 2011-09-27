/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/MPM/Contact/ContactFactory.h>
#include <CCA/Components/MPM/Contact/NullContact.h>
#include <CCA/Components/MPM/Contact/SingleVelContact.h>
#include <CCA/Components/MPM/Contact/FrictionContact.h>
#include <CCA/Components/MPM/Contact/NodalSVFContact.h>
#include <CCA/Components/MPM/Contact/SpecifiedBodyContact.h>
#include <CCA/Components/MPM/Contact/ApproachContact.h>
#include <CCA/Components/MPM/Contact/CompositeContact.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>

using namespace std;
using namespace Uintah;

Contact* ContactFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, SimulationStateP &ss,
                                MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");
   
   
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
      
      else if (con_type == "nodal_svf")
        contact_list->add(scinew NodalSVFContact(myworld,child,ss,lb,flag));
      
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
