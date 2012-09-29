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

#ifndef __NEO_HOOKEAN_SHEAR_H__
#define __NEO_HOOKEAN_SHEAR_H__

#include "ShearStressModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class NeoHookean
   *  \brief An isotropic neo Hookean shear stress model.
   *  \author Biswajit Banerjee, 
   *
  */
  class NeoHookean : public ShearStressModel {

  private:

    double d_mu;    // Shear modulus

    NeoHookean& operator=(const NeoHookean &smm);

  public:
         
    /*! Construct a neo Hookean shear stress model. */
    NeoHookean(ProblemSpecP& ps);

    /*! Construct a copy of neo Hookean shear stress model. */
    NeoHookean(const NeoHookean* smm);

    /*! Destructor of neo Hookean shear stress model.   */
    virtual ~NeoHookean();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the shear stress */
    void computeShearStress(const DeformationState* state, Matrix3& stress);
  };
} // End namespace Uintah
      
#endif  // __NEO_HOOKEAN_SHEAR_H__

