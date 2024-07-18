/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef ICE_VISCOSITY_SUTHERLAND_H
#define ICE_VISCOSITY_SUTHERLAND_H

#include <CCA/Components/ICE/ViscosityModel/Viscosity.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/MaterialManagerP.h>

namespace Uintah {

// Citation:
//    Cengel, Y., and Cimbala, J., "Fluid Mechanics Fundamentals and Applications, Third edition"
//    , McGraw-Hill Publishing, 2014, pg 53.

class Sutherland : public Viscosity {
public:
  Sutherland(ProblemSpecP& ps);
  ~Sutherland();

  virtual void outputProblemSpec(ProblemSpecP& ice_ps);

  //__________________________________
  //
  virtual void
  computeDynViscosity(CellIterator        & iter,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & cv)
  {
    computeDynViscosity_impl<CCVariable<double> >( iter, temp_CC, cv);
  }

  virtual void
  computeDynViscosity(CellIterator             & iter,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & cv)
  {
    computeDynViscosity_impl<constCCVariable<double> >( iter, temp_CC, cv);
  }


protected:

  double d_a;           // constants    air 1.458E-6 kg/(m s K^0.5) for air
  double d_b;           //              air 110.4 K

  template< class CCVar>
  void computeDynViscosity_impl( CellIterator      & iter,
                                 CCVar             & temp_CC,
                                 CCVariable<double>& cv);

};

}

#endif /* ICE_VISCOSITY_SUTHERLAND_H */

