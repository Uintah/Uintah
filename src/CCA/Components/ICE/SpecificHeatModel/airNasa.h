/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef _AIRNASA_H_
#define _AIRNASA_H_

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {

class airNasa : public SpecificHeat
{
public:

  airNasa( ProblemSpecP& ps );
  virtual ~airNasa();

  virtual void outputProblemSpec( ProblemSpecP& ice_ps );

  virtual double getGamma( double T );

  virtual double getInternalEnergy( double T );

  virtual void
  computeSpecificHeat( CellIterator&       iter,
                       CCVariable<double>& temp_CC,
                       CCVariable<double>& cv )
  {
    computeSpecificHeat_impl<CCVariable<double>>( iter, temp_CC, cv );
  }

  virtual void
  computeSpecificHeat( CellIterator&            iter,
                       constCCVariable<double>& temp_CC,
                       CCVariable<double>&      cv )
  {
    computeSpecificHeat_impl<constCCVariable<double>>( iter, temp_CC, cv );
  }

private:
  double Rair;
  const double d_Tmid = 1000; // Cutoff between high and low temperature polynomial
  const std::vector<double> d_aOxygenHighTemp = {
    3.28253784e+00, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14
  };
  const std::vector<double> d_aOxygenLowTemp = {
    3.78245636e+00, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12
  };
  const std::vector<double> d_aNitrogenHighTemp = {
    0.02926640e+02, 0.14879768e-02, -0.05684760e-05, 0.10097038e-09, -0.06753351e-13
  };
  const std::vector<double> d_aNitrogenLowTemp = {
    0.03298677e+02, 0.14082404e-02, -0.03963222E-04, 0.05641515e-07, -0.02444854e-10
  };

  const double d_xNitrogen = 3.76 / 4.76; // nitrogen mol fraction
  const double d_xOxygen   = 1.0 / 4.76; // oxygen mol fraction

  double computeCv_internal               ( double T ) const;
  double computeGamma_internal            ( double T ) const;
  double computeInternalEnergy_internal   ( double T ) const;

  template<class CCVar>
  void computeSpecificHeat_impl( CellIterator&       iter,
                                 CCVar&              temp_CC,
                                 CCVariable<double>& cv );
};

} // namespace Uintah

#endif /* _AIRNASA_H_ */
