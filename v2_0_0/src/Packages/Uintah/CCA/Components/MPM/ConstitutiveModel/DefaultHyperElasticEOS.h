#ifndef __DEFAULT_HYPERELASTIC_EOS_MODEL_H__
#define __DEFAULT_HYPERELASTIC_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   DefaultHyperElasticEOS
   
   Not really an equation of state but just an isotropic
   hypoelastic pressure calculator based on bulk and 
   shear moduli

GENERAL INFORMATION

   DefaultHyperElasticEOS.h

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

      class DefaultHyperElasticEOS : public MPMEquationOfState {

      // Create datatype for storing model parameters
      public:

      private:

	 // Prevent copying of this class
	 // copy constructor
	 DefaultHyperElasticEOS(const DefaultHyperElasticEOS &cm);
	 DefaultHyperElasticEOS& operator=(const DefaultHyperElasticEOS &cm);

      public:
	 // constructors
	 DefaultHyperElasticEOS(ProblemSpecP& ps); 
	 
	 // destructor 
	 virtual ~DefaultHyperElasticEOS();
	 
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

#endif  // __DEFAULT_HYPERELASTIC_EOS_MODEL_H__ 
