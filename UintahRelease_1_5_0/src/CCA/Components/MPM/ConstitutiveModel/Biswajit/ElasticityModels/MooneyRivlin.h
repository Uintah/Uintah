/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef __MOONEY_RIVLIN_SHEAR_H__
#define __MOONEY_RIVLIN_SHEAR_H__

#include "ShearStressModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class MooneyRivlin
   *  \brief An isotropic Mooney-Rivlin shear stress model.
   *  \author Biswajit Banerjee, 
   *
  */
  class MooneyRivlin : public ShearStressModel {

  private:

    double d_C1;    // Constant C_10  [mu = 2 (C_01 + C_10)]
    double d_C2;    // Constant C_01

    MooneyRivlin& operator=(const MooneyRivlin &smm);

  public:
         
    /*! Construct a Mooney-Rivlin shear stress model. */
    MooneyRivlin(ProblemSpecP& ps);

    /*! Construct a copy of Mooney-Rivlin shear stress model. */
    MooneyRivlin(const MooneyRivlin* smm);

    /*! Destructor of Mooney-Rivlin shear stress model.   */
    virtual ~MooneyRivlin();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the shear stress */
    void computeShearStress(const DeformationState* state, Matrix3& stress);
  };
} // End namespace Uintah
      
#endif  // __MOONEY_RIVLIN_SHEAR_H__

