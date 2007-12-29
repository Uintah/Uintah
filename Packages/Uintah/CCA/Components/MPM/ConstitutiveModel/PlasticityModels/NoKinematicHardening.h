#ifndef __NO_KINEMATIC_HARDENING_MODEL_H__
#define __NO_KINEMATIC_HARDENING_MODEL_H__


#include "KinematicHardeningModel.h"    
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class NoKinematicHardening
    \brief Default kinematic hardening model - no kinematic hardening
    \author Biswajit Banerjee, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2007 University of Utah
   
  */
  /////////////////////////////////////////////////////////////////////////////

  class NoKinematicHardening : public KinematicHardeningModel {

  private:

    // Prevent copying of this class
    // copy constructor
    //NoKinematicHardening(const NoKinematicHardening &cm);
    NoKinematicHardening& operator=(const NoKinematicHardening &cm);

  public:
    // constructors
    NoKinematicHardening();
    NoKinematicHardening(ProblemSpecP& ps);
    NoKinematicHardening(const NoKinematicHardening* cm);
         
    // destructor 
    virtual ~NoKinematicHardening();

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

#endif  // __NO_KINEMATIC_HARDENING_MODEL_H__ 
