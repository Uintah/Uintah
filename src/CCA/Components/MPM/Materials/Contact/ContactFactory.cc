/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/Materials/Contact/ContactFactory.h>
#include <CCA/Components/MPM/Materials/Contact/NullContact.h>
#include <CCA/Components/MPM/Materials/Contact/SingleVelContact.h>
#include <CCA/Components/MPM/Materials/Contact/FrictionContactBard.h>
#include <CCA/Components/MPM/Materials/Contact/FrictionContactLR.h>
#include <CCA/Components/MPM/Materials/Contact/FrictionContactLRVar.h>
#include <CCA/Components/MPM/Materials/Contact/NodalSVFContact.h>
#include <CCA/Components/MPM/Materials/Contact/SpecifiedBodyContact.h>
#include <CCA/Components/MPM/Materials/Contact/SpecifiedBodyFrictionContact.h>
#include <CCA/Components/MPM/Materials/Contact/ApproachContact.h>
#include <CCA/Components/MPM/Materials/Contact/CompositeContact.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>

using namespace std;
using namespace Uintah;

Contact* ContactFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, MaterialManagerP &ss,
                                MPMLabel* lb, MPMFlags* flag, bool &needNormals,
                                bool &useLogisticRegression)
{

   ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");

   if(!mpm_ps){
    string warn = "ERROR: Missing either <MaterialProperties> or <MPM> block from input file";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
   }
   
   CompositeContact * contact_list = scinew CompositeContact(myworld, lb, flag);
   
   needNormals=false;
   useLogisticRegression=false;

   for( ProblemSpecP child = mpm_ps->findBlock( "contact" ); child != nullptr; child = child->findNextBlock( "contact" ) ) {
     
     std::string con_type;
     child->getWithDefault("type",con_type, "null");
     
     if (con_type == "null") {
       contact_list->add(scinew NullContact(myworld,ss,lb,flag));
     }
     else if (con_type == "single_velocity") {
       contact_list->add(scinew SingleVelContact(myworld,child,ss,lb,flag));
     }
     else if (con_type == "nodal_svf") {
       contact_list->add(scinew NodalSVFContact(myworld,child,ss,lb,flag));
     }
     else if (con_type == "friction_LR") {
       contact_list->add(scinew FrictionContactLR(myworld,child,ss,lb,flag));
       useLogisticRegression=true;
     }
     else if (con_type == "friction_LRVar") {
       contact_list->add(scinew FrictionContactLRVar(myworld,child,ss,lb,flag));
       useLogisticRegression=true;
     }
     else if (con_type == "friction_bard") {
       contact_list->add(scinew FrictionContactBard(myworld,child,ss,lb,flag));
       needNormals=true;
     }
     else if (con_type == "approach") {
       contact_list->add(scinew ApproachContact(myworld,child,ss,lb,flag));
       needNormals=true;
     }
     else if (con_type == "specified_friction") {
       contact_list->add(scinew SpecifiedBodyFrictionContact(myworld,child,ss,lb,flag));
       useLogisticRegression=true;
     }
     else if (con_type == "specified_velocity" || con_type == "specified" || con_type == "rigid"  ) {
       contact_list->add( scinew SpecifiedBodyContact( myworld, child, ss, lb, flag ) );
       needNormals=true;
     }
     else {
       cerr << "Unknown Contact Type R (" << con_type << ")" << std::endl;;
       throw ProblemSetupException(" E R R O R----->MPM:Unknown Contact type", __FILE__, __LINE__);
     }
   }

   // 
   if( contact_list->size() == 0 ) {
     proc0cout << "no contact - using null\n";
     contact_list->add(scinew NullContact(myworld,ss,lb,flag));
   }

   return contact_list;
}
