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
#include "ShearModulus_Constant.h"
#include "ShearModulus_Nadal.h"
#include "ShearModulus_Borja.h"
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace std;
using namespace Uintah;

ShearModulusModel* ShearModulusModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("shear_modulus_model");
   if(!child) {
      cerr << "**WARNING** Creating default (constant shear modulus) model" << endl;
      return(scinew ShearModulus_Constant());
   }
   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("MPM::ConstitutiveModel:No type for shear modulus model.",
                                  __FILE__, __LINE__);
   
   if (mat_type == "constant_shear")
      return(scinew ShearModulus_Constant(child));
   else if (mat_type == "np_shear")
      return(scinew ShearModulus_Nadal(child));
   else if (mat_type == "borja")
      return(scinew ShearModulus_Borja(child));
   else {
      cerr << "**WARNING** Creating default (constant shear modulus) model" << endl;
      return(scinew ShearModulus_Constant(child));
   }
}

ShearModulusModel* 
ShearModulusModelFactory::createCopy(const ShearModulusModel* smm)
{
   if (dynamic_cast<const ShearModulus_Constant*>(smm))
      return(scinew ShearModulus_Constant(dynamic_cast<const ShearModulus_Constant*>(smm)));
   else if (dynamic_cast<const ShearModulus_Nadal*>(smm))
      return(scinew ShearModulus_Nadal(dynamic_cast<const ShearModulus_Nadal*>(smm)));
   else if (dynamic_cast<const ShearModulus_Borja*>(smm))
      return(scinew ShearModulus_Borja(dynamic_cast<const ShearModulus_Borja*>(smm)));
   else {
      cerr << "**WARNING** Creating copy of default (constant shear modulus) model" << endl;
      return(scinew ShearModulus_Constant(dynamic_cast<const ShearModulus_Constant*>(smm)));
   }
}
