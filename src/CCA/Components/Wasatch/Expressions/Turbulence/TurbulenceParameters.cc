/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

namespace WasatchCore {


  TurbulenceParameters::TurbulenceParameters()
  {
    turbSchmidt  = 1.0;
    turbPrandtl  = 0.7;
    eddyViscCoef = 0.1;
    turbModelName = TurbulenceParameters::NOTURBULENCE;
  };

  void parse_turbulence_input( Uintah::ProblemSpecP turbulenceInputParams,
                               TurbulenceParameters& turbParams )
  {
    if (!turbulenceInputParams) return;
    
    // get the name of the turbulence model
    std::string turbulenceModelName;
    turbulenceInputParams->getAttribute("model",turbulenceModelName);
    
    if ( turbulenceModelName.compare("SMAGORINSKY") == 0   ) {
      turbParams.turbModelName = TurbulenceParameters::SMAGORINSKY;
    } else if ( turbulenceModelName.compare("DYNAMIC") ==0 ) {    
      turbParams.turbModelName = TurbulenceParameters::DYNAMIC;
    } else if ( turbulenceModelName.compare("WALE")==0     ) {    
      turbParams.turbModelName = TurbulenceParameters::WALE;
    } else if ( turbulenceModelName.compare("VREMAN") == 0 ) {
      turbParams.turbModelName = TurbulenceParameters::VREMAN;
    } else {
      turbParams.turbModelName = TurbulenceParameters::NOTURBULENCE;
    }
    
    // get the eddy viscosity constant
    if ( turbParams.turbModelName != TurbulenceParameters::DYNAMIC )
      turbulenceInputParams->get("EddyViscosityCoefficient",turbParams.eddyViscCoef);

    // get the turbulent Schmidt Number
    turbulenceInputParams->getWithDefault("TurbulentSchmidt",turbParams.turbSchmidt, 0.7);

    // get the turbulent Prandtl number
    turbulenceInputParams->getWithDefault("TurbulentPrandtl",turbParams.turbPrandtl, 0.7 );
  }
  
}
