#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>

namespace Uintah {

  class Task;
  class Patch;
  class VarLabel;
  class ICEMaterial;

/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   EquationOfState.h

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

      class EquationOfState {
      public:
	 
	 EquationOfState();
	 virtual ~EquationOfState();
	 

	 
	 //////////
	 // Create space in data warehouse for CM data
	 virtual void initializeEOSData(const Patch* patch,
				       const ICEMaterial* matl,
				       DataWarehouseP& new_dw) = 0;

	 // C&R for sound speed calc.
	 virtual void addComputesAndRequiresSS(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const=0;

	 // C&R for rho micro calc.
	 virtual void addComputesAndRequiresRM(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const=0;

	 // C&R for press eos calc.
	 virtual void addComputesAndRequiresPEOS(Task* task,
					       const ICEMaterial* matl,
					       const Patch* patch,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw) const=0;

         virtual void computeSpeedSound(const Patch* patch,
					const ICEMaterial* matl,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw) = 0;

         virtual void computeRhoMicro(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) = 0;

	 // Per cell
         virtual double computeRhoMicro(double& press,double& gamma,
                                        double& cv, double& Temp) =0;

         virtual void computePressEOS(double& rhoM, double& gamma,
                                      double& cv, double& Temp,
                                      double& press, double& dp_drho, 
                                      double& dp_de) = 0;
         //per patch                          
        virtual void computeTempCC(const Patch* patch,
                                const CCVariable<double>& press, 
                                const double& gamma,
				const double& cv,
                                const CCVariable<double>& rho_micro, 
                                CCVariable<double>& Temp)=0;

	 
	 // Per patch
         virtual void computePressEOS(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) = 0;

	 void computeSpeedSoundMM(const Patch* patch,
				  const ICEMaterial* matl,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);


        protected:

	 ICELabel* lb;
      };
} // End namespace Uintah
      
#endif  // __EQUATION_OF_STATE_H__

