#ifndef __JOHNSONCOOK_DAMAGE_MODEL_H__
#define __JOHNSONCOOK_DAMAGE_MODEL_H__


#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

/**************************************

CLASS
   JohnsonCookDamage
   
   Rate dependent damage model 
   (Johnson and Cook, 1985, Int. J. Eng. Fracture Mech., 21, 31-48)

GENERAL INFORMATION

   JohnsonCookDamage.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2002 University of Utah

KEYWORDS
   Johnson-Cook, Viscoplasticity, Damage

DESCRIPTION
   
   The damage evolution rule is given by 

      d/dt(D) = d/dt(eps_p)/eps_pf

   where D = damage variable
            where D = 0 for virgin material
                  D = 1 for fracture
         eps_pf = value of fracture strain given by
         eps_pf = (D1 + D2 exp (D3 sigma*)][1+d/dt(p*)]^(D4)[1+D5 T*] 
            where sigma* = 1/3*trace(sigma)/sigma_eq
                  D1, D2, D3, D4, D5 are constants
         T* = (T-Troom)/(Tmelt-Troom)
             
WARNING
  
****************************************/

      class JohnsonCookDamage : public DamageModel {

      // Create datatype for storing model parameters
      public:
	 struct CMData {
            double D1;
            double D2;
            double D3;
            double D4;
            double D5;
            double TRoom;
            double TMelt;
	 };	 

      private:

	 CMData d_initialData;
	 
	 // Prevent copying of this class
	 // copy constructor
	 JohnsonCookDamage(const JohnsonCookDamage &cm);
	 JohnsonCookDamage& operator=(const JohnsonCookDamage &cm);

      public:
	 // constructors
	 JohnsonCookDamage(ProblemSpecP& ps); 
	 
	 // destructor 
	 virtual ~JohnsonCookDamage();
	 
	 //////////
	 // Calculate the scalar damage parameter 
	 virtual double computeScalarDamage(const Matrix3& rateOfDeformation,
                                            const Matrix3& stress,
                                            const double& temperature,
                                            const double& delT,
                                            const MPMMaterial* matl,
                                            const double& tolerance,
                                            const double& damage_old);
  
      protected:

         double calcStrainAtFracture(const Matrix3& sig, 
                                     const double& epdot,
                                     const double& T,
                                     const MPMMaterial* matl,
                                     const double& tolerance);
      };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_DAMAGE_MODEL_H__ 
