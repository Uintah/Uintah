#ifndef __HARLOW_H__
#define __HARLOW_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   IdealGas.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class Harlow : public EquationOfState {
      public:
	 
	 Harlow(ProblemSpecP& ps);
	 virtual ~Harlow();
	 
	 //////////
	 // Create space in data warehouse for CM data
	 virtual void initializeEOSData(const Patch* patch,
				       const ICEMaterial* matl,
				       DataWarehouseP& new_dw);

	 // C&R for sound speed calc.
	 virtual void addComputesAndRequiresSS(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const;

	 // C&R for rho micro calc.
	 virtual void addComputesAndRequiresRM(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const;
	 // C&R for press eos calc.
	 virtual void addComputesAndRequiresPEOS(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const;

	 // Per patch
         virtual void computeSpeedSound(const Patch* patch,
					const ICEMaterial* matl,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw);

         virtual void computePressEOS(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

	 virtual void computeRhoMicro(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

	 // Per cell
	 virtual double computeRhoMicro(double& press,double& gamma,
				        double& cv, double& Temp);
	 
	 virtual void computePressEOS(double& rhoM, double& gamma,
				      double& cv, double& Temp,
				      double& press, double& dp_drho,
				      double& dp_de);
                                  
       //per patch                          
        virtual void computeTempCC(const Patch* patch,
                                const CCVariable<double>& press, 
                                const double& gamma,
				    const double& cv,
                                const CCVariable<double>& rho_micro, 
                                CCVariable<double>& Temp);

        protected:

	 ICELabel* lb;
      };
} // End namespace Uintah
      
#endif  // __HARLOW_H__



