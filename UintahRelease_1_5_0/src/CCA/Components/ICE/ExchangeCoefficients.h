/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __EXCHANGE_COEFFICIENTS_H__
#define __EXCHANGE_COEFFICIENTS_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <vector>

namespace Uintah {
  class ExchangeCoefficients {
  public:
    ExchangeCoefficients();
    ~ExchangeCoefficients();

    void problemSetup(ProblemSpecP& ps,
                      SimulationStateP& sharedState);
    void outputProblemSpec(ProblemSpecP& ps);
    
    
    bool convective();
    int conv_fluid_matlindex();
    int conv_solid_matlindex();
    vector<double> K_mom();
    vector<double> K_heat();
    string d_heatExchCoeffModel;
    
  private:
    vector<double> d_K_mom, d_K_heat;
    bool d_convective;
    int d_conv_fluid_matlindex;
    int d_conv_solid_matlindex;
    
  };


}



#endif
