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


#include "ShearModulusModelFactory.h"
#include "ConstantShear.h"
#include "MTSShear.h"
#include "SCGShear.h"
#include "PTWShear.h"
#include "NPShear.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace std;
using namespace Uintah;

/// Create an instance of a Yield Condition.
/*! Available yield conditions are : von Mises, Gurson-Tvergaard-Needleman,
    Rosselier */
ShearModulusModel* ShearModulusModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("shear_modulus_model");
   if(!child) {
      proc0cout << "**WARNING** Creating default (constant shear modulus) model" << endl;
      return(scinew ConstantShear());
      //throw ProblemSetupException("MPM::ConstitutiveModel:Cannot find shear modulus model.",
      //                            __FILE__, __LINE__);
   }
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("MPM::ConstitutiveModel:No type for shear modulus model.",
                                  __FILE__, __LINE__);
   
   if (mat_type == "constant_shear")
      return(scinew ConstantShear(child));
   else if (mat_type == "mts_shear")
      return(scinew MTSShear(child));
   else if (mat_type == "scg_shear")
      return(scinew SCGShear(child));
   else if (mat_type == "ptw_shear")
      return(scinew PTWShear(child));
   else if (mat_type == "np_shear")
      return(scinew NPShear(child));
   else {
      proc0cout << "**WARNING** Creating default (constant shear modulus) model" << endl;
      return(scinew ConstantShear(child));
      //throw ProblemSetupException("MPM::ConstitutiveModel:Unknown Shear Modulus Model ("+mat_type+")",
      //                            __FILE__, __LINE__);
   }
}

ShearModulusModel* 
ShearModulusModelFactory::createCopy(const ShearModulusModel* smm)
{
   if (dynamic_cast<const ConstantShear*>(smm))
      return(scinew ConstantShear(dynamic_cast<const ConstantShear*>(smm)));
   else if (dynamic_cast<const MTSShear*>(smm))
      return(scinew MTSShear(dynamic_cast<const MTSShear*>(smm)));
   else if (dynamic_cast<const SCGShear*>(smm))
      return(scinew SCGShear(dynamic_cast<const SCGShear*>(smm)));
   else if (dynamic_cast<const PTWShear*>(smm))
      return(scinew PTWShear(dynamic_cast<const PTWShear*>(smm)));
   else if (dynamic_cast<const NPShear*>(smm))
      return(scinew NPShear(dynamic_cast<const NPShear*>(smm)));
   else {
      proc0cout << "**WARNING** Creating copy of default (constant shear modulus) model" << endl;
      return(scinew ConstantShear(dynamic_cast<const ConstantShear*>(smm)));
      //throw ProblemSetupException("Cannot create copy of unknown shear modulus model",
      //                            __FILE__, __LINE__);
   }
}
