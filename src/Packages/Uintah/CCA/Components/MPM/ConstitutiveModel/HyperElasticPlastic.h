#ifndef __HYPERELASTICPLASTIC_MODEL_H__
#define __HYPERELASTICPLASTIC_MODEL_H__


#include "ConstitutiveModel.h"	
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMEquationOfState.h>
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {

/**************************************

CLASS
   HyperElasticPlastic
   
   Hyperelastic plastic constitutive model for isotropic materials

GENERAL INFORMATION

   HyperElasticPlastic.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Hyperelastic Plastic, Viscoplasticity

DESCRIPTION
   
   Return mapping plasticity algorithm based on a multiplicative 
   decomposition of the deformation gradient and the intermediate
   configuration concept.

   Algorithm taken from :

   Simo and Hughes, 1998, Computational Inelasticity, p. 319.
  
WARNING
  
   Isotropic materials, J2 plasticity

****************************************/

  class HyperElasticPlastic : public ConstitutiveModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;
      double Shear;
    };	 

    // Left Cauchy-Green tensor based on the deviatoric part
    // of the elastic part of the deformation gradient 
    const VarLabel* pBbarElasticLabel;  
    const VarLabel* pBbarElasticLabel_preReloc;

    // Scalar damage evolution variable
    const VarLabel* pDamageLabel;  
    const VarLabel* pDamageLabel_preReloc;  

  private:

    CMData d_initialData;
	 
    bool d_useMPMICEModifiedEOS;
    double d_tol;
    double d_damageCutOff;
    PlasticityModel* d_plasticity;
    DamageModel* d_damage;
    MPMEquationOfState* d_eos;

    // Prevent copying of this class
    // copy constructor
    HyperElasticPlastic(const HyperElasticPlastic &cm);
    HyperElasticPlastic& operator=(const HyperElasticPlastic &cm);

  public:

    // constructors
    HyperElasticPlastic(ProblemSpecP& ps, MPMLabel* lb,int n8or27);
	 
    // destructor 
    virtual ~HyperElasticPlastic();
	 
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);


    virtual void allocateCMData(DataWarehouse* new_dw,
				ParticleSubset* subset,
				map<const VarLabel*, ParticleVariableBase*>* newState);


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

    //////////
    // Sockets for MPM-ICE
    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    //////////
    // Sockets for MPM-ICE
    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    //////////
    // Sockets for MPM-ICE
    virtual double getCompressibility();

  };
} // End namespace Uintah

#endif  //__HYPERELASTICPLASTIC_MODEL_H__
