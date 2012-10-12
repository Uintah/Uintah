/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "LinearMeltTemp.h"
#include <Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;
         
// Construct a melt temp model.  
LinearMeltTemp::LinearMeltTemp( )
{
  d_useVolumeForm   = false;
  d_usePressureForm = false;
}

LinearMeltTemp::LinearMeltTemp(ProblemSpecP& ps)
{
  ps->require(       "T_m0",   d_Tm0);
  ps->getWithDefault("a",      d_a,    -999.0);
  ps->getWithDefault("b",      d_b,    -999.0);
  ps->getWithDefault("Gamma_0",d_Gamma,-999.0);
  ps->getWithDefault("K_T",    d_K_T,  -999.0);

  // If 'a' is defined and 'K_T' is defined, then use b
  if(d_a != -999.0 && d_K_T != -999.0 ) {
    d_usePressureForm = true;
    d_b = d_a*d_Tm0/d_K_T;
    return;
  }

  // If 'b' is specified, everything is fine
  if(d_b != -999.0) {
    d_usePressureForm = true;
  }

  // If 'a' is specified, everything is fine
  if(d_a != -999.0) {
   d_useVolumeForm = true;
   return;
  }

  // If 'a' and 'b' are not specified by 'Gamma' is,
  //  use Lindemann Law (see Poirier, J.-P. Introduction 
  //     to the Physics of the Earth's Interior, Cambridge
  //      Univ. Press, Cambridge, UK, 1991.)
  if(d_a == -999.0 && d_b == -999.0 && d_Gamma != -999.0) {
    d_a = 2.0*(d_Gamma-1.0/3.0);
    return;
  }

  throw ProblemSetupException("ERROR: In Linear Melting Temperature Model, no good parameters specified.",__FILE__, __LINE__);
}

// Construct a copy of a melt temp model.  
LinearMeltTemp::LinearMeltTemp(const LinearMeltTemp* lhs )
{
  this->d_Tm0   = lhs->d_Tm0;
  this->d_a     = lhs->d_a;
  this->d_b     = lhs->d_b;
  this->d_Gamma = lhs->d_Gamma;
  this->d_K_T   = lhs->d_K_T;
}

// Destructor of melt temp model.  
LinearMeltTemp::~LinearMeltTemp()
{
}

void LinearMeltTemp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP temp_ps = ps->appendChild("melting_temp_model");
  temp_ps->setAttribute("type","linear_Tm");
  
  temp_ps->appendElement("T_m0",    d_Tm0);
  temp_ps->appendElement("a",      d_a);
  temp_ps->appendElement("b",      d_b);
  temp_ps->appendElement("Gamma_0",d_Gamma);
  temp_ps->appendElement("K_T",  d_K_T);
}

         
// Compute the melt temp
double 
LinearMeltTemp::computeMeltingTemp(const PlasticityState* state)
{
  // Pressure form takes presidence because it is less computationally expensive (no divide)
  if(d_usePressureForm) {
     return d_Tm0 + d_b * state->pressure;
  } else if(d_useVolumeForm) {
     return d_Tm0*(1.0+d_a*(state->initialDensity/state->density));
  }
  return d_Tm0;
}

