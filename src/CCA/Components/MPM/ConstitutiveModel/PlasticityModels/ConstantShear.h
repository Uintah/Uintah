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

#ifndef __CONSTANT_SHEAR_MODEL_H__
#define __CONSTANT_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class ConstantShear
   *  \brief The shear modulus does not vary with density and temperature
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *
  */
  class ConstantShear : public ShearModulusModel {

  private:
    ConstantShear& operator=(const ConstantShear &smm);

  public:
         
    /*! Construct a constant shear modulus model. */
    ConstantShear();
    ConstantShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    ConstantShear(const ConstantShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~ConstantShear();
         
    virtual void outputProblemSpec(ProblemSpecP& ps);

    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __CONSTANT_SHEAR_MODEL_H__

