#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <vector>
#include <Packages/Uintah/Core/Math/Sparse.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/Grid/Array3.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscsles.h"
}
#endif

#define MAX_BASIS 27



namespace Uintah {

  class Task;
  class Patch;
  class VarLabel;
  class MPMLabel;
  class MPMMaterial;
  class DataWarehouse;


/**************************************

CLASS
   ConstitutiveModel
   
   Short description...

GENERAL INFORMATION

   ConstitutiveModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Constitutive_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class ConstitutiveModel {
      public:
	 
	 ConstitutiveModel();
	 virtual ~ConstitutiveModel();
	 
	 //////////
	 // Basic constitutive model calculations
	 virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw) = 0;

	 virtual void computeStressTensorImplicit(const PatchSubset* patches,
						  const MPMMaterial* matl,
						  DataWarehouse* old_dw,
						  DataWarehouse* new_dw,
						  SparseMatrix<double,int>& K,
#ifdef HAVE_PETSC
						  Mat &A,
						  map<const Patch*, Array3<int> >& d_petscLocalToGlobal,

#endif
						  const bool recursion);
	 
	 virtual void computeStressTensorImplicitOnly(const PatchSubset* patches,
						  const MPMMaterial* matl,
						  DataWarehouse* old_dw,
						      DataWarehouse* new_dw);
	 

	 //////////
	 // Create space in data warehouse for CM data
	 virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw) = 0;

	 virtual void addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const = 0;

	 virtual void addComputesAndRequires(Task* task,
					   const MPMMaterial* matl,
					   const PatchSet* patches) const = 0;

	 virtual void addComputesAndRequiresImplicit(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patches,
					    const bool recursion);

	 virtual void addComputesAndRequiresImplicitOnly(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patches,
					    const bool recursion);

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to) = 0;

         virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
					  const MPMMaterial* matl) = 0;

         virtual void computePressEOSCM(double rho_m, double& press_eos,
                                        double p_ref,
                                        double& dp_drho, double& ss_new,
	        		        const MPMMaterial* matl) = 0;

         virtual double getCompressibility() = 0;

	 double computeRhoMicro(double press,double gamma,
				        double cv, double Temp);
	 
	 void computePressEOS(double rhoM, double gamma,
				      double cv, double Temp,
				      double& press, double& dp_drho,
				      double& dp_de);

        protected:

	 MPMLabel* lb;
         int d_8or27;
         int NGP;
         int NGN;
      };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__

