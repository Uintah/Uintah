#ifndef __MTS_PLASTICITY_MODEL_H__
#define __MTS_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   MTSPlastic
   
   Mechanical Threshold Stress Internal Variable Plasticity Model
   (Folansbee, P. S. and Kocks, U. F, 1988, Acta Metallurgica, 36(1), 81-93.
    Folansbee, P. S., 1989, Mechanical Properties of Materials at High Rates
                            of Strain, IOP Conference, pp. 213-220.) 

GENERAL INFORMATION

   MTSPlastic.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Mechanical Threshold Stress Model, Viscoplasticity

DESCRIPTION
   
   The flow stress is given by

      sigma = s_a + s_t[1 - {kT ln(edot0/edot)/(g0 mu b^3)}^(1/q)]^(1/p)

   where sigma = equivalent flow stress
         s_a = athermal component of mechanical threshold stress = (for Cu) 40 MPa
         k = Boltzmann constant = 1.38e-23 J/K
         b = Burger's vector length = (for Cu) 2.55e-10 m
         k/b^3 = 0.823 MPa/K (for Cu)
         edot0 = Gamma0 = constant = (for Cu) 1.0e7 /s
         g0 = normalized activation energy = (for annealed Cu) 1.6
         q = constant = (for Cu) 1
         p = constant = (for Cu) 2/3
         T = absolute temperature
         edot = strain rate (evolves according to balance laws)
         mu = shear modulus (evolves with temperature)
            = b1 - b2/(exp(b3/T) - 1)

         s_t = s - s_a = thermal component of mechanical threshold stress (evolves)
         s = total mechanical threshold stress
         ds/de = strain hardening rate = theta = theta_0 [ 1 - F(X)]
                 where theta_0 = hardening due to dislocation accumulation
                               = a0 + a1 ln(edot) + a2 edot
                                 where
                                     a0, a1, a2 = constants
                       X = (s - s_a)/(s_s - s_a)
                           where s_s = stress at zero strain hardening rate
                                     = s_s0 (edot/edot_s0)]^[kT/(mu b^3 A)]
                                       where s_s0 = saturation threshold stress
                                                    for deformation at 0 K 
                                                  = (for Cu) 770MPa
                                             A = constant = (for Cu) 0.2625
                                             edot_s0 = const = 1e7 /s
                       F = tanh(alpha X)/tanh(alpha) , 
                           where alpha = 2 for Cu.
         
    Internal Variable Evolution :
      
     s(n+1) = s(n) + delEps*theta

WARNING
  
****************************************/

      class MTSPlastic : public PlasticityModel {

	 // Create datatype for storing model parameters
      public:
	 struct CMData {
            double s_a;
            double koverbcubed;
            double edot0;
            double g0;
            double q;
            double p;
            double alpha;
            double edot_s0;
            double A;
            double s_s0;
            double a0;
            double a1;
            double a2;
            double b1;
            double b2;
            double b3;
            double mu_0;
	 };	 
         const VarLabel* pMTSLabel; // For MTS model
         const VarLabel* pMTSLabel_preReloc; // For MTS model

      private:

	 CMData d_const;
	 
	 // Prevent copying of this class
	 // copy constructor
	 MTSPlastic(const MTSPlastic &cm);
	 MTSPlastic& operator=(const MTSPlastic &cm);

      public:
	 // constructors
	 MTSPlastic(ProblemSpecP& ps);
	 
	 // destructor 
	 virtual ~MTSPlastic();
	 
         // Computes and requires for internal evolution variables
         // Only one internal variable for MTS model :: mechanical threshold stress
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
      };

} // End namespace Uintah

#endif  // __MTS_PLASTICITY_MODEL_H__ 
