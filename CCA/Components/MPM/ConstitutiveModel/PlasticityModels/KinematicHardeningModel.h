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

#ifndef __KINEMATIC_HARDENING_MODEL_H__
#define __KINEMATIC_HARDENING_MODEL_H__

#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityState.h"


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  KinematicHardeningModel
    \brief  Abstract Base class for kinematic hardening models 
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah,\n
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
    /*! \brief Calculate the back stress */
    /* Note that df_dsigma_normal_new is the normalized value of df_dsigma */
    //////////
    virtual void computeBackStress(const PlasticityState* state,
                                   const double& delT,
                                   const particleIndex idx,
                                   const double& delLambda,
                                   const Matrix3& df_dsigma_normal_new,
                                   const Matrix3& backStress_old,
                                   Matrix3& backStress_new) = 0;
 
    /*! Compute the direction of back stress evolution (\f$h^beta\f$) 
        for the equation \f$ d/dt(\beta) = d/dt(\gamma) h^beta \f$ */
    virtual void eval_h_beta(const Matrix3& df_dsigma,
                             const PlasticityState* state,
                             Matrix3& h_beta) = 0;

    /*! Get the back stress */
    void getBackStress(const particleIndex idx,
                       Matrix3& backStress);

    /*! Update the back stress */
    void updateBackStress(const particleIndex idx,
                          const Matrix3& backStress);

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

