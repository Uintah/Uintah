#ifndef __DEFAULT_HYPOELASTIC_EOS_MODEL_H__
#define __DEFAULT_HYPOELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   DefaultHypoElasticEOS
   
   Not really an equation of state but just an isotropic
   hypoelastic pressure calculator based on bulk and 
   shear moduli

GENERAL INFORMATION

   DefaultHypoElasticEOS.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2003 University of Utah

KEYWORDS
   Isotropic hypoelastic model, Pressure calculation

DESCRIPTION
   
   The equation of state is given by

       p = -(Trace(D) * lambda * delT)

   where 
         p = pressure
         I = identity matrix
         D = rate of deformation tensor
         lambda = K - 2/3*mu
             where K = bulk modulus
                   mu = shear modulus
         delT = time increment
             
WARNING
  
****************************************/

      class DefaultHypoElasticEOS : public MPMEquationOfState {

      // Create datatype for storing model parameters
      public:

      private:

	 // Prevent copying of this class
	 // copy constructor
	 DefaultHypoElasticEOS(const DefaultHypoElasticEOS &cm);
	 DefaultHypoElasticEOS& operator=(const DefaultHypoElasticEOS &cm);

      public:
	 // constructors
	 DefaultHypoElasticEOS(ProblemSpecP& ps); 
	 
	 // destructor 
	 virtual ~DefaultHypoElasticEOS();
	 
	 //////////
	 // Calculate the pressure using a equation of state
	 virtual Matrix3 computePressure(const MPMMaterial* matl,
                                        const double& bulk,
                                        const double& shear,
                                        const Matrix3& deformGrad,
                                        const Matrix3& rateOfDeformation,
                                        const Matrix3& stress,
                                        const double& temperature,
                                        const double& density,
                                        const double& delT);
  
      };

} // End namespace Uintah

#endif  // __DEFAULT_HYPOELASTIC_EOS_MODEL_H__ 
