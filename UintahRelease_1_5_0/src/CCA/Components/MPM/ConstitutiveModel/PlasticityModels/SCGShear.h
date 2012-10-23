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

#ifndef __SCG_SHEAR_MODEL_H__
#define __SCG_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class SCGShear
   *  \brief The shear modulus model used by Steinberg,Cochran,Guinan in
   *         the SCG plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *
  */
  class SCGShear : public ShearModulusModel {

  private:

    double d_mu0; // Material constant (also in SCG model)
    double d_A;   // Material constant (also in SCG model)
    double d_B;   // Material constant (also in SCG model)

    SCGShear& operator=(const SCGShear &smm);

  public:
         
    /*! Construct a constant shear modulus model. */
    SCGShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    SCGShear(const SCGShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~SCGShear();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __SCG_SHEAR_MODEL_H__

