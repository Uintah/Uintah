/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef __DAMAGE_MODEL_H__
#define __DAMAGE_MODEL_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>


namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class DamageModel
    \brief Abstract base class for damage models   
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class DamageModel {
  public:
         
    DamageModel();
    virtual ~DamageModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Initialize the damage parameter in the calling function
    */
    //////////////////////////////////////////////////////////////////////////
    virtual double initialize() = 0;

    //////////////////////////////////////////////////////////////////////////
    /*! 
      Determine if damage has crossed cut off
    */
    //////////////////////////////////////////////////////////////////////////
    virtual bool hasFailed(double damage) = 0;
    
    //////////////////////////////////////////////////////////////////////////
    /*! 
      Calculate the scalar damage parameter 
    */
    //////////////////////////////////////////////////////////////////////////
    virtual double computeScalarDamage(const double& plasticStrainRate,
                                       const Matrix3& stress,
                                       const double& temperature,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const double& tolerance,
                                       const double& damage_old) = 0;

  };
} // End namespace Uintah
      


#endif  // __DAMAGE_MODEL_H__

