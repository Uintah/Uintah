#ifndef __VISCO_PLASTICITY_MODEL_H__
#define __VISCO_PLASTICITY_MODEL_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
// #include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
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
    \class  PlasticityModel
    \brief  Abstract Base class for plasticity models (calculate yield stress)
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah,\n
            Copyright (C) 2002 University of Utah\n
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
//    			    Matrix3& reducedEta, 
// 			    double& xae, 
// 			    const particleIndex idx,
// 			    const Matrix3 pStress);
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

