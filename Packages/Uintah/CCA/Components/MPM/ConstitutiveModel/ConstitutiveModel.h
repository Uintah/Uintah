#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
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
  class ParticleSubset;
  class ParticleVariableBase;


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

    virtual void allocateCMData(DataWarehouse* new_dw,
				ParticleSubset* subset,
				map<const VarLabel*, ParticleVariableBase*>* newState) = 0;

    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const=0;

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

    /////////
    // Add initial computes for erosion
    virtual void addInitialComputesAndRequiresWithErosion(Task* task,
				     const MPMMaterial* matl,
				     const PatchSet* patches,
				     std::string algorithm);

    //////////
    // Computes and requires to erosion update
    virtual void addComputesAndRequiresWithErosion(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patch) const;

    //////////
    // Stress computation with erosion
    virtual void computeStressTensorWithErosion(const PatchSubset* patches,
				const MPMMaterial* matl,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

    //////////
    // Carry forward CM variables for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
			      const MPMMaterial* matl,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

    //////////
    // Carry forward CM variables for RigidMPM with Erosion
    virtual void carryForwardWithErosion(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    //////////
    // Convert J-integral into stress intensity factors for hypoelastic materials 
    virtual void ConvertJToK(const MPMMaterial* matl,const Vector& J,
                             const Vector& C,const Vector& V,Vector& SIF);

  protected:

    // Calculate velocity gradient for 27 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    const Vector& psize, 
				    constNCVariable<Vector>& gVelocity);
    // for dual veocity field
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx,
                                    const Point& px,
                                    const Vector& psize,
                                    const short pgFld[], 
                                    constNCVariable<Vector>& gVelocity,
                                    constNCVariable<Vector>& GVelocity);
 
    // Calculate velocity gradient for 8 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    constNCVariable<Vector>& gVelocity);
    // for dual velocity field 
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx,
                                    const Point& px,
                                    const short pgFld[],
                                    constNCVariable<Vector>& gVelocity,
                                    constNCVariable<Vector>& GVelocity);

    // Calculate polar decomposition
    void polarDecomposition(const Matrix3& F, 
                            Matrix3& R,
                            Matrix3& U) const;

    MPMLabel* lb;
    int d_8or27;
    int NGP;
    int NGN;
    std::string d_erosionAlgorithm;
  };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__

