/*
 * Copyright (c) 2012 The University of Utah
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

#include "TurbulenceParameters.h"

namespace Wasatch {

  void parse_turbulence_input(Uintah::ProblemSpecP turbulenceInputParams,
                              TurbulenceParameters& turbParams)
  {
    if (!turbulenceInputParams) return;
    
    // get the name of the turbulence model
    std::string turbulenceModelName;
    turbulenceInputParams->get("TurbulenceModel",turbulenceModelName);    
    if ( turbulenceModelName.compare("SMAGORINSKY") == 0   ) {
      turbParams.turbulenceModelName = SMAGORINSKY;
    } else if ( turbulenceModelName.compare("DYNAMIC") ==0 ) {    
      turbParams.turbulenceModelName = DYNAMIC;  
    } else if ( turbulenceModelName.compare("WALE")==0     ) {    
      turbParams.turbulenceModelName = WALE;
    } else if ( turbulenceModelName.compare("VREMAN") == 0 ) {
      turbParams.turbulenceModelName = VREMAN;
    } else {
      turbParams.turbulenceModelName = NONE;
    }
    
    // get the eddy viscosity constant
    turbulenceInputParams->getWithDefault("EddyViscosityConstant",turbParams.eddyViscosityConstant, 0.1);
    
    // get the kolmogorov scale
    turbulenceInputParams->getWithDefault("KolmogorovScale",turbParams.kolmogorovScale, 1e100);    

    // get the turbulent Schmidt Number
    turbulenceInputParams->getWithDefault("TurbulentSchmidt",turbParams.turbulentSchmidt, 1.0);      
  }
  
}