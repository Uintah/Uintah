#ifndef __PRAGER_KINEMATIC_HARDENING_MODEL_H__
#define __PRAGER_KINEMATIC_HARDENING_MODEL_H__


#include "KinematicHardeningModel.h"    
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class PragerKinematicHardening
    \brief Default kinematic hardening model - no kinematic hardening
    \author Biswajit Banerjee, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2007 University of Utah
   
    The kinematic hardening rule is given by
    \f[
    \dot{\beta} = \frac{2}{3}~H_prime~d_p
    \f]
    where \f$\beta\f$ is the back stress, \f$H_prime\f$ is the hardening 
    modulus, and \f$ d_p\f$ is the plastic rate of deformation.

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
      d_p = \sqrt{\frac{3}{2}}~\dot{\lambda}~n
    \f]
    Therefore, the evolution equation for beta can be written as
    \f[
    \dot{\beta} = \sqrt{\frac{2}{3}}~H_1~\dot{\lambda}~n
    \f]
    A backward Euler discretization leads to
    \f[
    \beta_{n+1} = \beta_n + \sqrt{\frac{2}{3}}~H_1~\Delta\lambda~\n_{n+1}
    \f]
  */
  /////////////////////////////////////////////////////////////////////////////

  class PragerKinematicHardening : public KinematicHardeningModel {

  protected:

    struct CMData {
      double beta;  // beta is a parameter between 0 and 1
                    // 0 == no kinematic hardening
      double hardening_modulus; // the kinematic hardening modulus
    };

  private:

    CMData d_cm;

    // Prevent copying of this class
    // copy constructor
    //PragerKinematicHardening(const PragerKinematicHardening &cm);
    PragerKinematicHardening& operator=(const PragerKinematicHardening &cm);

  public:
    // constructors
    PragerKinematicHardening(ProblemSpecP& ps);
    PragerKinematicHardening(const PragerKinematicHardening* cm);
         
    // destructor 
    virtual ~PragerKinematicHardening();

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

  };

} // End namespace Uintah

#endif  // __PRAGER_KINEMATIC_HARDENING_MODEL_H__ 
