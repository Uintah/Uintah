#ifndef __IDEAL_GAS_H__
#define __IDEAL_GAS_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Components/ICE/ICELabel.h>
#include "EquationOfState.h"

namespace Uintah {
   class Task;
   class Patch;
   class VarLabel;
   namespace ICESpace {
      class ICEMaterial;

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

      class IdealGas : public EquationOfState {
      public:
	 
	 IdealGas(ProblemSpecP& ps);
	 virtual ~IdealGas();
	 
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


         virtual void computeSpeedSound(const Patch* patch,
                                        const ICEMaterial* matl,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw);
	 // Per patch
         virtual void computePressEOS(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

	 virtual void computeRhoMicro(const Patch* patch,
				      const ICEMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

	 // Per cell
	 virtual double computeRhoMicro(double& ,double& );
	 
	 virtual double computePressEOS(double&, double&);

        protected:

	 ICELabel* lb;
      };
      
   } // end namespace ICE
} // end namespace Uintah
#endif  // __IDEAL_GAS_H__

// $Log$
// Revision 1.2  2000/10/10 20:35:12  jas
// Move some stuff around.
//
// Revision 1.1  2000/10/06 04:02:16  jas
// Move into a separate EOS directory.
//
// Revision 1.2  2000/10/05 04:26:48  guilkey
// Added code for part of the EOS evaluation.
//
// Revision 1.1  2000/10/04 23:40:12  jas
// The skeleton framework for an EOS model.  Does nothing.
//

