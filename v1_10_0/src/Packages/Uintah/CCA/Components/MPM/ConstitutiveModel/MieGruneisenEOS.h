#ifndef __MIE_GRUNEISEN_EOS_MODEL_H__
#define __MIE_GRUNEISEN_EOS_MODEL_H__


#include "MPMEquationOfState.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   MieGruneisenEOS
   
   A Mie-Gruneisen type equation of state model
   (Zocher, Maudlin, Chen, Flower-Maudlin, 2000,
    European Congress on Computational Methods in Applied Science 
    and Engineering, ECOMAS 2000, Barcelona)

GENERAL INFORMATION

   MieGruneisenEOS.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2003 University of Utah

KEYWORDS
   Mie-Gruneisen, Equation of State, Pressure calculation

DESCRIPTION
   
   The equation of state is given by

       p = (rho_0 * (C_0)^2 * zeta * (1 + (1-Gamma_0/2)*zeta))/
           (1 - (S_alpha - 1)*zeta)^2 + Gamma_0*E

   where 
         p = pressure
         C_0 = bulk speed of sound
         zeta = (rho/rho_0 - 1)
              where rho = current density
                    rho_0 = initial density
         E = internal energy = c_p * T
              where c_p = specfic heat at constant pressure
                    T = temperature
         Gamma_0 = Gruneisen's gamma at reference state
         S_alpha = linear Hugoniot slope coefficient
             
WARNING
  
****************************************/

      class MieGruneisenEOS : public MPMEquationOfState {

      // Create datatype for storing model parameters
      public:
	 struct CMData {
            double C_0;
            double Gamma_0;
            double S_alpha;
	 };	 

      private:

	 CMData d_const;
	 
	 // Prevent copying of this class
	 // copy constructor
	 MieGruneisenEOS(const MieGruneisenEOS &cm);
	 MieGruneisenEOS& operator=(const MieGruneisenEOS &cm);

      public:
	 // constructors
	 MieGruneisenEOS(ProblemSpecP& ps); 
	 
	 // destructor 
	 virtual ~MieGruneisenEOS();
	 
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

#endif  // __MIE_GRUNEISEN_EOS_MODEL_H__ 
