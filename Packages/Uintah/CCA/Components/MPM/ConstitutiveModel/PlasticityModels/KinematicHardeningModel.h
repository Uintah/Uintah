#ifndef __KINEMATIC_HARDENING_MODEL_H__
#define __KINEMATIC_HARDENING_MODEL_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityState.h"


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  KinematicHardeningModel
    \brief  Abstract Base class for kinematic hardening models 
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah,\n
            Copyright (C) 2007 University of Utah\n
    \warn   Assumes vonMises yield condition and the associated flow rule for 
            all cases other than Gurson plasticity.
  */
  ///////////////////////////////////////////////////////////////////////////

  class KinematicHardeningModel {

  public:

    constParticleVariable<Matrix3> pBackStress;
    ParticleVariable<Matrix3> pBackStress_new;

    const VarLabel* pBackStressLabel;
    const VarLabel* pBackStressLabel_preReloc;

  public:
         
    KinematicHardeningModel();
    virtual ~KinematicHardeningModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    //////////
    /*! \brief Calculate the kinematic hardening modulus */
    //////////
    virtual double computeKinematicHardeningModulus(const PlasticityState* state,
                                                    const double& delT,
                                                    const MPMMaterial* matl,
                                                    const particleIndex idx) = 0;
 
    //////////
    /*! \brief Calculate the back stress */
    //////////
    virtual void computeBackStress(const PlasticityState* state,
                                   const double& delT,
                                   const particleIndex idx,
                                   const double& delGamma,
                                   const Matrix3& df_dsigma_new,
                                   Matrix3& backStress_new) = 0;
 
    /*!  Data management apparatus */
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse) const;


    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual void initializeBackStress(ParticleSubset* pset,
                                      DataWarehouse* new_dw);

    virtual void getBackStress(ParticleSubset* pset,
                               DataWarehouse* old_dw);

    virtual void allocateAndPutBackStress(ParticleSubset* pset,
                                          DataWarehouse* new_dw); 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw); 

  };
} // End namespace Uintah
      


#endif  // __KINEMATIC_HARDENING_MODEL_H__

