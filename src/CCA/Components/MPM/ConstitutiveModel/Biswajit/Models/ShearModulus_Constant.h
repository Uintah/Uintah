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

#ifndef __BB_CONSTANT_SHEAR_MODEL_H__
#define __BB_CONSTANT_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace UintahBB {

  /*! \class ShearModulus_Constant
   *  \brief The shear modulus does not vary with density and temperature
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *
  */
  class ShearModulus_Constant : public ShearModulusModel {

  private:
    ShearModulus_Constant& operator=(const ShearModulus_Constant &smm);

  public:
         
    /*! Construct a constant shear modulus model. */
    ShearModulus_Constant();
    ShearModulus_Constant(Uintah::ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    ShearModulus_Constant(const ShearModulus_Constant* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~ShearModulus_Constant();
         
    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps);

    /*! Compute the shear modulus */
    double computeInitialShearModulus();
    double computeShearModulus(const ModelState* state);
    double computeShearModulus(const ModelState* state) const;

    /*! Compute the shear strain energy */
    double computeStrainEnergy(const ModelState* state) {return 0.0;};

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute q = 3 mu epse_s
         where mu = shear modulus
               epse_s = sqrt{2/3} ||ee||
               ee = deviatoric part of elastic strain = epse - 1/3 epse_v I
               epse = total elastic strain
               epse_v = tr(epse)
    */
    /////////////////////////////////////////////////////////////////////////
    double computeQ(const ModelState* state) const
    {
      return 0.0;
    };

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute dq/depse_s 
    */
    /////////////////////////////////////////////////////////////////////////
    double computeDqDepse_s(const ModelState* state) const
    {
      return 0.0;
    };

    /////////////////////////////////////////////////////////////////////////
    /* 
      Compute dq/depse_v 
    */
    /////////////////////////////////////////////////////////////////////////
    double computeDqDepse_v(const ModelState* state) const
    {
      return 0.0;
    };
  };
} // End namespace Uintah
      
#endif  // __CONSTANT_SHEAR_MODEL_H__

