#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {
  class MPMLabel;

  /**************************************
CLASS
   CompMooneyRivlin
   
   Short description...

GENERAL INFORMATION

   CompMooneyRivlin.h

   Author?
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Comp_Mooney_Rivlin

DESCRIPTION
   Long description...
  
WARNING
  
  ****************************************/

  class CompMooneyRivlin : public ConstitutiveModel {
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double C1;
      double C2;
      double PR;
    };
  private:
    CMData d_initialData;
	 
    // Prevent copying of this class
    // copy constructor
    CompMooneyRivlin(const CompMooneyRivlin &cm);
    CompMooneyRivlin& operator=(const CompMooneyRivlin &cm);

  public:
    // constructor
    CompMooneyRivlin(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
	 
    // destructor 
    virtual ~CompMooneyRivlin();
	 
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

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);
	 
    virtual double getCompressibility();

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
					   const PatchSet* patch, 
					   MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* addset,
				   map<const VarLabel*, ParticleVariableBase*>* newState,
				   ParticleSubset* delset,
				   DataWarehouse* old_dw);

	 
    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);
  };
} // End namespace Uintah


#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

