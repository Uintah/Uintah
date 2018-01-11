/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/MPM/Dissolution/DissolutionFactory.h>
#include <CCA/Components/MPM/Dissolution/NullDissolution.h>
#include <CCA/Components/MPM/Dissolution/TestDissolution.h>
#include <CCA/Components/MPM/Dissolution/StressRateDissolution.h>
#include <CCA/Components/MPM/Dissolution/CompositeDissolution.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <string>

using namespace std;
using namespace Uintah;

Dissolution* DissolutionFactory::create(const ProcessorGroup* myworld,
                                const ProblemSpecP& ps, SimulationStateP &ss,
                                MPMLabel* lb, MPMFlags* flag)
{

   ProblemSpecP mpm_ps = 
     ps->findBlockWithOutAttribute("MaterialProperties")->findBlock("MPM");

   if(!mpm_ps){
    string warn = "ERROR: Missing either <MaterialProperties> or <MPM> block from input file";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
   }
   
   CompositeDissolution * dissolution_list = scinew CompositeDissolution(myworld, lb, flag);

   for( ProblemSpecP child = mpm_ps->findBlock( "dissolution" ); 
                     child != nullptr; 
                     child = child->findNextBlock( "dissolution" ) ) {
     
     std::string dis_type;
     child->getWithDefault("type",dis_type, "null");
     
     if (dis_type == "null") {
      dissolution_list->add(scinew NullDissolution(myworld,ss,lb,flag));
     }
//     else if (dis_type == "stress_threshold") {
     else if (dis_type == "test") {
      dissolution_list->add(scinew TestDissolution(myworld,child,ss,lb,flag));
      flag->d_doingDissolution=true;
     }
     else if (dis_type == "stress_rate") {
      dissolution_list->add(scinew StressRateDissolution(myworld,
                                                         child,ss,lb,flag));
      flag->d_doingDissolution=true;
     }
     else {
       cerr << "Unknown Dissolution Type R (" << dis_type << ")" << std::endl;;
       throw ProblemSetupException(" E R R O R----->MPM:Unknown Dissolution type", __FILE__, __LINE__);
     }
   }

   // 
   if( dissolution_list->size() == 0 ) {
     proc0cout << "no dissolution - using null\n";
     dissolution_list->add(scinew NullDissolution(myworld,ss,lb,flag));
   }

   return dissolution_list;
}
