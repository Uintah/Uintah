#ifndef __PLASTICITY_MODEL_H__
#define __PLASTICITY_MODEL_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>


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

  class PlasticityModel {

  private:

  public:
	 
    PlasticityModel();
    virtual ~PlasticityModel();
	 
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) 
                                               const = 0;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const = 0;

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
					   const PatchSet* patch, 
					   MPMLabel* lb) const = 0;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* addset,
				   map<const VarLabel*, ParticleVariableBase*>* newState,
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

    virtual void updatePlastic(const particleIndex idx, 
                               const double& delGamma) = 0;

    virtual double getUpdatedPlasticStrain(const particleIndex idx) = 0;

    //////////
    // Calculate the flow stress
    virtual double computeFlowStress(const Matrix3& rateOfDeformation,
				     const double& temperature,
				     const double& delT,
				     const double& tolerance,
				     const MPMMaterial* matl,
				     const particleIndex idx) = 0;
 
    /*! Compute the elastic-plastic tangent modulus 
      This is given by
      \f[ 
      C_{ep} = C_e - \frac{(C_e:r) (x) (f_s:C_e)}
      {-f_q.h + f_s:C_e:r}
      \f]
      where \n
      \f$ C_{ep} \f$ is the continuum elasto-plastic tangent modulus \n
      \f$ C_{e} \f$ is the continuum elastic tangent modulus \n
      \f$ r \f$ is the plastic flow direction \f$ d\phi/d\sigma = r \f$\n
      \f$ h \f$ gives the evolution of \f$ q \f$ \n
      \f$ f_s = \partial f /\partial \sigma \f$ \n
      \f$ f_q = \partial f /\partial q \f$ 
    */
    virtual void computeTangentModulus(const Matrix3& stress,
				       const Matrix3& rateOfDeform, 
				       double temperature,
				       double delT,
                                       const particleIndex idx,
                                       const MPMMaterial* matl,
				       TangentModulusTensor& Ce,
				       TangentModulusTensor& Cep) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to scalar and
        internal variables.

      \return Three derivatives in Vector derivs 
        (derivs[0] = \f$d\sigma_Y/d\dot\epsilon\f$,
         derivs[1] = \f$d\sigma_Y/dT\f$, 
         derivs[2] = \f$d\sigma_Y/d(int. var.)\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual void evalDerivativeWRTScalarVars(double edot,
                                             double T,
                                             const particleIndex idx,
                                             Vector& derivs) = 0;

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic
        strain.

      \return \f$d\sigma_Y/d\epsilon\f$
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double evalDerivativeWRTPlasticStrain(double edot, double T,
                                                  const particleIndex idx) = 0;
  };
} // End namespace Uintah
      


#endif  // __PLASTICITY_MODEL_H__

