#ifndef __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__
#define __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__


#include "KinematicHardeningModel.h"    
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ArmstrongFrederickKinematicHardening
    \brief Default kinematic hardening model - no kinematic hardening
    \author Biswajit Banerjee, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2007 University of Utah
   
    The kinematic hardening rule is given by
    \f[
    \dot{\beta} = \frac{2}{3}~H_1~d_p - H_2~\beta~||d_p||
    \f]
    where \f$\beta\f$ is the back stress, \f$H_1, H_2\f$ are the two hardening 
    moduli, and \f$ d_p\f$ is the plastic rate of deformation.

    For associative plasticity
    \f[
      d_p = dot{\gamma}~\frac{\partial f}{\partial sigma}
    \f]
    For von Mises plasticity
    \f[
      \frac{\partial f}{\partial sigma} = \frac{\sigma-\beta}{||\sigma-\beta||} = n
    \f]
    and
    \f[
      ||d_p|| = dot{gamma}
    \f]
    Therefore, the evolution equation for beta can be written as
    \f[
    \dot{\beta} = \frac{2}{3}~H_1~\dot{gamma}~n - H_2~\beta~\dot{gamma}
    \f]
    A backward Euler discretization leads to
    \f[
    \beta_{n+1} - \beta_n = \Delta\gamma(\frac{2}{3}~H_1~n_{n+1} - H_2~\beta_{n+1})
    \f]
    or
    \f[
    \beta_{n+1} = \frac{1}{1 + H_2~\Delta\gamma}(\beta_n + \frac{2}{3}\Delta\gamma~H_1~n_{n+1})
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
    /*! \brief Calculate the kinematic hardening modulus */
    //////////
    virtual double computeKinematicHardeningModulus(const PlasticityState* state,
                                     const double& delT,
                                     const MPMMaterial* matl,
                                     const particleIndex idx);
 
    //////////
    /*! \brief Calculate the back stress */
    //////////
    virtual void computeBackStress(const PlasticityState* state,
                                   const double& delT,
                                   const particleIndex idx,
                                   const double& delGamma,
                                   const Matrix3& df_dsigma_new,
                                   Matrix3& backStress_new);

  };

} // End namespace Uintah

#endif  // __ARMSTRONG_FREDERICK_KINEMATIC_HARDENING_MODEL_H__ 
