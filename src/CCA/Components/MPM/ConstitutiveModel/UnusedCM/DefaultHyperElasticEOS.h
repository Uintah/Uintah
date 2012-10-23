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

#ifndef __DEFAULT_HYPERELASTIC_EOS_MODEL_H__
#define __DEFAULT_HYPERELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h" 
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class DefaultHyperElasticEOS
    \brief Not really an equation of state but just an isotropic
    hyperelastic pressure calculator based on bulk modulus
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n

    The equation of state is given by
    \f[
    p = 0.5 K (J - \frac{1}{J})
    \f]
    where,\n
    \f$ K \f$ is the bulk modulus \n
    \f$ J \f$ is the Jacobian
  */
  ////////////////////////////////////////////////////////////////////////////

  class DefaultHyperElasticEOS : public MPMEquationOfState {

    // Create datatype for storing model parameters
  public:

  private:

    // Prevent copying of this class
    // copy constructor
    //DefaultHyperElasticEOS(const DefaultHyperElasticEOS &cm);
    DefaultHyperElasticEOS& operator=(const DefaultHyperElasticEOS &cm);

  public:
    // constructors
    DefaultHyperElasticEOS(ProblemSpecP& ps); 
    DefaultHyperElasticEOS(const DefaultHyperElasticEOS* cm);
         
    // destructor 
    virtual ~DefaultHyperElasticEOS();
         
    //////////
    // Calculate the pressure using a equation of state
    double computePressure(const MPMMaterial* matl,
                           const PlasticityState* state,
                           const Matrix3& deformGrad,
                           const Matrix3& rateOfDeformation,
                           const double& delT);
  
  };

} // End namespace Uintah

#endif  // __DEFAULT_HYPERELASTIC_EOS_MODEL_H__ 
