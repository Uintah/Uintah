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


#include "YieldConditionFactory.h"
#include "VonMisesYield.h"
#include "GursonYield.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace std;
using namespace Uintah;

/// Create an instance of a Yield Condition.
/*! Available yield conditions are : von Mises, Gurson-Tvergaard-Needleman,
    Rosselier */
YieldCondition* YieldConditionFactory::create(ProblemSpecP& ps, const bool usingRR)
{
  ProblemSpecP child = ps->findBlock("yield_condition");
  if(!child)
    throw ProblemSetupException("MPM::ConstitutiveModel:Cannot find yield condition.", __FILE__, __LINE__);

  string mat_type;
  if(!child->getAttribute("type", mat_type))
    throw ProblemSetupException("MPM::ConstitutiveModel:No type for yield condition.", __FILE__, __LINE__);

  if (mat_type == "vonMises")
    return(scinew VonMisesYield(child));
  else if (mat_type == "gurson"){
    if( usingRR ){
      ostringstream warn;
      warn << "MPM::ConstitutiveModel:Yield Condition ("+mat_type+")"
           << " only works with the 'biswajit' plastic convergence algorithm\n"
           << " Add: \n"
           << "    <plastic_convergence_algo> biswajit </plastic_convergence_algo> \n"
           << " to your input file ";
           
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    return(scinew GursonYield(child));
  }
  else 
    throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Yield Condition ("+mat_type+")",
                                 __FILE__, __LINE__);
}

YieldCondition* 
YieldConditionFactory::createCopy(const YieldCondition* yc)
{
   if (dynamic_cast<const VonMisesYield*>(yc))
      return(scinew VonMisesYield(dynamic_cast<const VonMisesYield*>(yc)));

   else if (dynamic_cast<const GursonYield*>(yc))
      return(scinew GursonYield(dynamic_cast<const GursonYield*>(yc)));

   else 
      throw ProblemSetupException("Cannot create copy of unknown yield condition", __FILE__, __LINE__);
}
