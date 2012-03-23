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


#include "ShearModulus_Constant.h"

using namespace Uintah;
         
// Construct a shear modulus model.  
ShearModulus_Constant::ShearModulus_Constant()
{
}

ShearModulus_Constant::ShearModulus_Constant(ProblemSpecP& )
{
}

// Construct a copy of a shear modulus model.  
ShearModulus_Constant::ShearModulus_Constant(const ShearModulus_Constant* )
{
}

// Destructor of shear modulus model.  
ShearModulus_Constant::~ShearModulus_Constant()
{
}


void ShearModulus_Constant::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP shear_ps = ps->appendChild("shear_modulus_model");
  shear_ps->setAttribute("type","constant_shear");
}
         
// Compute the shear modulus
double 
ShearModulus_Constant::computeInitialShearModulus()
{
  return state->initialShearModulus;
}

double 
ShearModulus_Constant::computeShearModulus(const ModelState* state)
{
  return state->initialShearModulus;
}

