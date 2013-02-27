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

#ifndef __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__
#define __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__


#include "KinematicHardeningModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ArmstrongFrederickKinematicHardening
    \brief Default kinematic hardening model - no kinematic hardening
    \author Biswajit Banerjee, 
    Department of Mechanical Engineering, 
    University of Utah
   
    The kinematic hardening rule is given by
    \f[
    \dot{\beta} = \frac{2}{3}~H_1~d_p - H_2~\beta~||d_p||
    \f]
    where \f$\beta\f$ is the back stress, \f$H_1, H_2\f$ are the two hardening 
    moduli, and \f$ d_p\f$ is the plastic rate of deformation.

    For associative plasticity
    \f[
      d_p = dot{\lambda}~\frac{\partial f}{\partial sigma}
    \f]
    For von Mises plasticity with the yield condition of the form
    \f[
      f := \sqrt{\frac{3}{2} \xi:\xi} - \sigma_y \le 0
    \f]
    where \f$\xi = s - \beta\f$ and \f$s\f$ is the deviatoric part of the Cauchy 
    stress, we have
    \f[
      \frac{\partial f}{\partial sigma} = \sqrt{\frac{3}{2}}\cfrac{\xi}{||\xi||} 
       = \sqrt{\frac{3}{2}}~n ~;~~ ||n|| = 1
    \f]
    and
    \f[
      d_p = \sqrt{\frac{3}{2}}~\dot{\lambda}~n; ||d_p|| = \sqrt{\frac{3}{2}}~\dot{\lambda}
    \f]
    Therefore, the evolution equation for beta can be written as
    \f[
    \dot{\beta} = \sqrt{\frac{2}{3}}~H_1~\dot{\lambda}~n - \sqrt{\frac{3}{2}}~H_2~\beta~\dot{\lambda}
    \f]
    A backward Euler discretization leads to
    \f[
    \beta_{n+1} - \beta_n = \Delta\lambda(\sqrt{\frac{2}{3}}~H_1~/n_{n+1} - \sqrt{\frac{3}{2}}~H_2~\beta_{n+1})
    \f]
    or
    \f[
    \beta_{n+1} = \frac{1}{1 + \sqrt{\frac{3}{2}}~H_2~\Delta\lambda}(\beta_n + \sqrt{\frac{2}{3}}~\Delta\lambda~H_1~n_{n+1})
    \f]
  */
  /////////////////////////////////////////////////////////////////////////////

  class ArmstrongFrederickKinematicHardening : public KinematicHardeningModel {

  protected:

    struct CMData {
      double beta;
      double hardening_modulus_1; // H_1 in the model
      double hardening_modulus_2; // H_2 in the model
    };

  private:

    CMData d_cm;

    // Prevent copying of this class
    // copy constructor
    //ArmstrongFrederickKinematicHardening(const ArmstrongFrederickKinematicHardening &cm);
    ArmstrongFrederickKinematicHardening& operator=(const ArmstrongFrederickKinematicHardening &cm);

  public:
    // constructors
    ArmstrongFrederickKinematicHardening(ProblemSpecP& ps);
    ArmstrongFrederickKinematicHardening(const ArmstrongFrederickKinematicHardening* cm);
         
    // destructor 
    virtual ~ArmstrongFrederickKinematicHardening();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    //////////
    /*! \brief Calculate the back stress */
    //////////
    virtual void computeBackStress(const PlasticityState* state,
                                   const double& delT,
                                   const particleIndex idx,
                                   const double& delLambda,
                                   const Matrix3& df_dsigma_new,
                                   const Matrix3& backStress_old,
                                   Matrix3& backStress_new);

    void eval_h_beta(const Matrix3& df_dsigma,
                     const PlasticityState* state,
                     Matrix3& h_beta);
  };

} // End namespace Uintah

#endif  // __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__ 
