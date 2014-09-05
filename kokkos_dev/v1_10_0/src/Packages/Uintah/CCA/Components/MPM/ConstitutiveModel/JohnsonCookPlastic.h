#ifndef __JOHNSONCOOK_PLASTICITY_MODEL_H__
#define __JOHNSONCOOK_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   JohnsonCookPlastic
   
   Strain rate dependent plasticity model
   (Johnson and Cook, 1983, Proc. 7th Intl. Symp. Ballistics, The Hague)

GENERAL INFORMATION

   JohnsonCookPlastic.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Johnson-Cook, Viscoplasticity

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

WARNING
  
****************************************/

      class JohnsonCookPlastic : public PlasticityModel {

	 // Create datatype for storing model parameters
      public:
	 struct CMData {
            double A;
            double B;
            double C;
            double n;
            double m;
            double TRoom;
            double TMelt;
	 };	 

         constParticleVariable<double> pPlasticStrain;
         ParticleVariable<double> pPlasticStrain_new;

         const VarLabel* pPlasticStrainLabel;  // For Johnson-Cook Plasticity
         const VarLabel* pPlasticStrainLabel_preReloc;  // For Johnson-Cook Plasticity

      private:

	 CMData d_initialData;
         
	 // Prevent copying of this class
	 // copy constructor
	 JohnsonCookPlastic(const JohnsonCookPlastic &cm);
	 JohnsonCookPlastic& operator=(const JohnsonCookPlastic &cm);

      public:
	 // constructors
	 JohnsonCookPlastic(ProblemSpecP& ps);
	 
	 // destructor 
	 virtual ~JohnsonCookPlastic();
	 
         // Computes and requires for internal evolution variables
         // Only one internal variable for Johnson-Cook :: plastic strain
	 virtual void addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patches) const;

	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const PatchSet* patches) const;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);

         virtual void initializeInternalVars(ParticleSubset* pset,
				             DataWarehouse* new_dw);

         virtual void getInternalVars(ParticleSubset* pset,
                                      DataWarehouse* old_dw);

         virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                                 DataWarehouse* new_dw); 

         virtual void updateElastic(const particleIndex idx);

         virtual void updatePlastic(const particleIndex idx, const double& delGamma);

	 // compute the flow stress
         virtual double computeFlowStress(const Matrix3& rateOfDeformation,
                                          const Matrix3& stress,
                                          const double& temperature,
                                          const double& delT,
                                          const double& tolerance,
                                          const MPMMaterial* matl,
                                          const particleIndex idx);

      protected:

	 double evaluateFlowStress(const double& ep, 
				   const double& epdot,
				   const double& T,
                                   const MPMMaterial* matl,
				   const double& tolerance);

      };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_PLASTICITY_MODEL_H__ 
