#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <vector>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/CCA/Components/MPM/Solver.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>


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
				     DataWarehouse* new_dw);

    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     Solver* solver,
				     const bool recursion);
	 
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
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

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

    // Calculate velocity gradient for 27 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    const Vector& psize, 
				    constNCVariable<Vector>& gVelocity);

    // Calculate velocity gradient for 8 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    constNCVariable<Vector>& gVelocity);

    MPMLabel* lb;
    int d_8or27;
    int NGP;
    int NGN;
  };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__

