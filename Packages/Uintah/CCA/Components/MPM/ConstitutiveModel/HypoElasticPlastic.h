#ifndef __HYPOELASTIC_PLASTIC_H__
#define __HYPOELASTIC_PLASTIC_H__


#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/YieldCondition.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/StabilityCheck.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMEquationOfState.h>
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {

/**************************************

CLASS
   HypoElasticPlastic
   
   General Hypo-Elastic Plastic Constitutive Model

GENERAL INFORMATION

   HypoElasticPlastic.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Hypo-elastic Plastic, Viscoplasticity

DESCRIPTION
   
   The rate of deformation and stress is rotated to material configuration before
   the updated values are calculated.  The left stretch and rotation are updated
   incrementatlly to get the deformation gradient.

   The flow rule can be any appropriate flow rule that is determined by a derived
   class, for example, 1) Johnson-Cook 2) Bammann 3) MTS

WARNING
  
   Only isotropic materials, von-Mises plasticity, associated flow rule,
   high strain rate.

****************************************/

  class HypoElasticPlastic : public ConstitutiveModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;
      double Shear;
    };	 

    const VarLabel* pLeftStretchLabel;  // For Hypoelastic-plasticity
    const VarLabel* pLeftStretchLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pDamageLabel;  // For Hypoelastic-plasticity
    const VarLabel* pDamageLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pPlasticTempLabel;  // For Hypoelastic-plasticity
    const VarLabel* pPlasticTempLabel_preReloc;  // For Hypoelastic-plasticity

  private:

    CMData d_initialData;

    double d_tol;
    double d_damageCutOff;
    bool d_useModifiedEOS;
    YieldCondition* d_yield;
    StabilityCheck* d_stable;
    PlasticityModel* d_plasticity;
    DamageModel* d_damage;
    MPMEquationOfState* d_eos;
	 
    // Prevent copying of this class
    // copy constructor
    HypoElasticPlastic(const HypoElasticPlastic &cm);
    HypoElasticPlastic& operator=(const HypoElasticPlastic &cm);

  public:

    // constructors
    HypoElasticPlastic(ProblemSpecP& ps, MPMLabel* lb,int n8or27);
	 
    // destructor 
    virtual ~HypoElasticPlastic();
	 
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     Solver* solver,
				     const bool recursion);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    virtual void computeStressTensorWithErosion(const PatchSubset* patches,
				const MPMMaterial* matl,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

    /////////
    // Sockets for MPM-ICE
    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    /////////
    // Sockets for MPM-ICE
    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    /////////
    // Sockets for MPM-ICE
    virtual double getCompressibility();

    // class function to read correct number of parameters
    // from the input file
    static void readParameters(ProblemSpecP ps, double *p_array);

    // class function to write correct number of parameters
    // from the input file, and create a new object
    static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);

    // member function to read correct number of parameters
    // from the input file, and any other particle information
    // need to restart the model for this particle
    // and create a new object
    static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps);

    // class function to create a new object from parameters
    static ConstitutiveModel* create(double *p_array);
  
  protected:

    // Compute the updated left stretch and rotation tensors
    void computeUpdatedVR(const double& delT,
			  const Matrix3& DD, 
			  const Matrix3& WW,
			  Matrix3& VV, 
			  Matrix3& RR);  

    // Compute the rate of rotation tensor
    Matrix3 computeRateofRotation(const Matrix3& tensorV, 
				  const Matrix3& tensorD,
				  const Matrix3& tensorW);

    /*! Compute the elastic tangent modulus tensor for isotropic
        materials */
    void computeElasticTangentModulus(double bulk,
                                      double shear,
                                      TangentModulusTensor& Ce);

  };

} // End namespace Uintah

#endif  // __HYPOELASTIC_PLASTIC_H__ 
