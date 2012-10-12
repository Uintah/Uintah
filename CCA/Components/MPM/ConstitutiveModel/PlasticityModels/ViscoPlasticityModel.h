/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __VISCO_PLASTICITY_MODEL_H__
#define __VISCO_PLASTICITY_MODEL_H__

#include <Core/Math/Matrix3.h>
#include <vector>
// #include <Core/Grid/Variables/ParticleSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/TangentModulusTensor.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "PlasticityState.h"


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  PlasticityModel
    \brief  Abstract Base class for plasticity models (calculate yield stress)
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah,\n
    \warn   Assumes vonMises yield condition and the associated flow rule for 
            all cases other than Gurson plasticity.
  */
  ///////////////////////////////////////////////////////////////////////////

  class ViscoPlasticityModel {

  private:

  public:
         
    ViscoPlasticityModel();
    virtual ~ViscoPlasticityModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) 
                                               const = 0;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const = 0;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse) const = 0;

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const = 0;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw) = 0;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to) = 0;

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw) = 0;

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw) = 0;

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw) = 0; 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw) = 0; 

    virtual void updateElastic(const particleIndex idx) = 0;

    virtual double computeFlowStress(
                                     const particleIndex idx,
                                     const Matrix3 pStress,
                                     const Matrix3 tensorR,
                                     const int implicitFlag) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the shear modulus. 
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeShearModulus(const PlasticityState* state) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the melting temperature
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeMeltingTemp(const PlasticityState* state) = 0;
    
//    virtual void computeNij(Matrix3& nij, 
//                          Matrix3& reducedEta, 
//                          double& xae, 
//                          const particleIndex idx,
//                          const Matrix3 pStress);
//                          
   virtual void computeStressIncTangent(double& epdot,
                                         Matrix3& stressRate, 
                                         TangentModulusTensor& Cep,
                                         const double delT,
                                         const particleIndex idx, 
                                         const TangentModulusTensor Ce,
                                         const Matrix3 tensorD,
                                         const Matrix3 pStress,
                                         const int implicitFlag,
                                         const Matrix3 tensorR)=0;
                                         
   //returns true if maximum tensile stress is exceeded                                  
   virtual bool checkFailureMaxTensileStress(const Matrix3 pStress)=0;
   
  };
} // End namespace Uintah
      


#endif  // __VISCO_PLASTICITY_MODEL_H__

