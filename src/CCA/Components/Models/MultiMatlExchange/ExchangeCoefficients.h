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

#ifndef __EXCHANGE_COEFFICIENTS_H__
#define __EXCHANGE_COEFFICIENTS_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Math/FastMatrix.h>

#include <vector>

namespace Uintah {
namespace ExchangeModels{

  class ExchangeCoefficients {
  public:
    ExchangeCoefficients();
    ~ExchangeCoefficients();

    void problemSetup( const ProblemSpecP  & ps,
                       const int numMatls,
                       ProblemSpecP        & exch_ps );

    void outputProblemSpec(ProblemSpecP& matl_ps,
                           ProblemSpecP& exch_prop_ps);

    void getConstantExchangeCoeff( FastMatrix& K,
                                   FastMatrix& H );

    void getVariableExchangeCoeff( FastMatrix& ,
                                   FastMatrix& H,
                                   IntVector & c,
                                   std::vector<constCCVariable<double> >& mass );
    bool convective();
    int conv_fluid_matlindex();
    int conv_solid_matlindex();
    std::string d_heatExchCoeffModel;

  private:
    std::vector<double> d_K_mom_V;
    std::vector<double> d_K_heat_V;
    bool d_convective = false;
    int d_conv_fluid_matlindex = -9;
    int d_conv_solid_matlindex = -9;
    int d_numMatls = 0;
  };
}
}

#endif
