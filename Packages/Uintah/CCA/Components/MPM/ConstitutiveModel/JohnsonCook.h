#ifndef __JOHNSONCOOK_CONSTITUTIVE_MODEL_H__
#define __JOHNSONCOOK_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {
  class TypeDescription;
  /**************************************

CLASS
   JohnsonCook
   
   Rate dependent viscoplasticity model with damage 
   (Johnson and Cook, 1983, Proc. 7th Intl. Symp. Ballistics, The Hague)
   (Johnson and Cook, 1985, Int. J. Eng. Fracture Mech., 21, 31-48)

GENERAL INFORMATION

   JohnsonCook.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Johnson-Cook, Viscoplasticity, Damage

DESCRIPTION
   
   The flow rule is given by

      f(sigma) = [A + B (eps_p)^n][1 + C ln(d/dt(eps_p*))][1 - (T*)^m]

   where f(sigma) = equivalent stress
         eps_p = plastic strain
         d/dt(eps_p*) = d/dt(eps_p)/d/dt(eps_p0) 
            where d/dt(eps_p0) = a user defined plastic strain rate
         A, B, C, n, m are material constants
            (for HY-100 steel tubes :
             A = 316 MPa, B = 1067 MPa, C = 0.0277, n = 0.107, m = 0.7)
         A is interpreted as the initial yield stress - sigma_0
         T* = (T-Troom)/(Tmelt-Troom)

   The damage evolution rule is given by 

      d/dt(D) = d/dt(eps_p)/eps_pf

   where D = damage variable
            where D = 0 for virgin material
                  D = 1 for fracture
         eps_pf = value of fracture strain given by
         eps_pf = (D1 + D2 exp (D3 sigma*)][1+d/dt(p*)]^(D4)[1+D5 T*] 
            where sigma* = 1/3*trace(sigma)/sigma_eq
                  D1, D2, D3, D4, D5 are constants
             
WARNING
  
  ****************************************/

  class JohnsonCook : public ConstitutiveModel {
    // Create datatype for storing model parameters
  private:

  public:
    struct CMData {
      double Bulk;
      double Shear;
      double A;
      double B;
      double C;
      double n;
      double m;
      double TRoom;
      double TMelt;
      double D1;
      double D2;
      double D3;
      double D4;
      double D5;
    };	 

    const VarLabel* pLeftStretchLabel;  // For Hypoelastic-plasticity
    const VarLabel* pLeftStretchLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pDeformRatePlasticLabel;  // For Hypoelastic-plasticity
    const VarLabel* pDeformRatePlasticLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pPlasticStrainLabel;  // For Hypoelastic-plasticity
    const VarLabel* pPlasticStrainLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pDamageLabel;  // For Hypoelastic-plasticity
    const VarLabel* pDamageLabel_preReloc;  // For Hypoelastic-plasticity

  private:
    friend const TypeDescription* fun_getTypeDescription(CMData*);

    CMData d_initialData;
	 
    // Prevent copying of this class
    // copy constructor
    JohnsonCook(const JohnsonCook &cm);
    JohnsonCook& operator=(const JohnsonCook &cm);

  public:
    // constructors
    JohnsonCook(ProblemSpecP& ps, MPMLabel* lb,int n8or27);
	 
    // destructor 
    virtual ~JohnsonCook();
	 
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

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    virtual double getCompressibility();

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

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
    static ConstitutiveModel* readRestartParametersAndCreate(
							     ProblemSpecP ps);

    // class function to create a new object from parameters
    static ConstitutiveModel* create(double *p_array);
  
  protected:

    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    const Vector& psize, 
				    constNCVariable<Vector>& gVelocity);

    Matrix3 computeVelocityGradient(const Patch* patch,
				    const double* oodx, 
				    const Point& px, 
				    constNCVariable<Vector>& gVelocity);

    void computeUpdatedVR(const double& delT,
			  const Matrix3& DD, 
			  const Matrix3& WW,
			  Matrix3& VV, 
			  Matrix3& RR);  

    Matrix3 computeRateofRotation(const Matrix3& tensorV, 
				  const Matrix3& tensorD,
				  const Matrix3& tensorW);

    double evaluateFlowStress(const double& ep, 
			      const double& epdot,
			      const double& T);

    double calcStrainAtFracture(const Matrix3& sig, 
				const double& epdot,
				const double& T);
  };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_CONSTITUTIVE_MODEL_H__ 
