/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/MPM/Materials/Dissolution/DissolutionFactory.h>
#include <CCA/Components/MPM/Materials/Dissolution/NullDissolution.h>
#include <CCA/Components/MPM/Materials/Dissolution/ContactStressIndependent.h>
#include <CCA/Components/MPM/Materials/Dissolution/ContactStressDependent.h>
#include <CCA/Components/MPM/Materials/Dissolution/SaltPrecipitationModel.h>
#include <CCA/Components/MPM/Materials/Dissolution/QuartzOvergrowth.h>
#include <CCA/Components/MPM/Materials/Dissolution/NewQuartzOvergrowth.h>
#include <CCA/Components/MPM/Materials/Dissolution/CompositeDissolution.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>

using namespace std;
using namespace Uintah;

Dissolution* DissolutionFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, MaterialManagerP &ss,
                                MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");

   if(!mpm_ps){
    string warn = "ERROR: Missing either <MaterialProperties> or <MPM> block from input file";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
   }
   
   CompositeDissolution * dissolution_list = scinew CompositeDissolution(myworld,lb);

   for( ProblemSpecP child = mpm_ps->findBlock( "dissolution" ); 
                     child != nullptr; 
                     child = child->findNextBlock( "dissolution" ) ) {
     
     std::string dis_type;
     child->getWithDefault("type",dis_type, "null");
     
     if (dis_type == "null") {
      dissolution_list->add(scinew NullDissolution(myworld,ss,lb));
      flag->d_doingDissolution=false;
      flag->d_computeNormals=false;
     }
     else if (dis_type == "contactStressIndependent") {
      dissolution_list->add(scinew ContactStressIndependent(myworld,child,ss,lb));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else if (dis_type == "contactStressDependent") {
      dissolution_list->add(scinew ContactStressDependent(myworld,child,ss,lb));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else if (dis_type == "saltPrecipitationModel") {
      dissolution_list->add(scinew SaltPrecipitationModel(myworld,child,ss,lb));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else if (dis_type == "QuartzOvergrowth") {
      dissolution_list->add(scinew QuartzOvergrowth(myworld,child,ss,lb));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else if (dis_type == "NewQuartzOvergrowth") {
      dissolution_list->add(scinew NewQuartzOvergrowth(myworld,child,ss,lb));
      flag->d_doingDissolution=true;
      flag->d_computeNormals=true;
     }
     else {
       cerr << "Unknown Dissolution Type R (" << dis_type << ")" << std::endl;;
       throw ProblemSetupException(" ERROR----->MPM:Unknown Dissolution type",
                                     __FILE__, __LINE__);
     }
   }

   // 
   if( dissolution_list->size() == 0 ) {
     proc0cout << "no dissolution - using null\n";
     dissolution_list->add(scinew NullDissolution(myworld,ss,lb));
   }

   return dissolution_list;
}
