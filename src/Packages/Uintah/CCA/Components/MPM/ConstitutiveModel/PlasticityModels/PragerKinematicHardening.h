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

#endif  // __PRAGER_KINEMATIC_HARDENING_MODEL_H__ 
