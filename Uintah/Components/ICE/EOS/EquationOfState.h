#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Components/ICE/ICELabel.h>

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

	 // C&R for compute equilibration pressure 
	 virtual void addComputesAndRequiresCEB(Task* task,
					        const ICEMaterial* matl,
					        const Patch* patch,
					        DataWarehouseP& old_dw,
					        DataWarehouseP& new_dw) const=0;

         virtual void computeSpeedSound(const Patch* patch,
                                 const ICEMaterial* matl,
                                 DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw) = 0;

         virtual void computeEquilibrationPressure(const Patch* patch,
	                                           const ICEMaterial* matl,
                 	                           DataWarehouseP& old_dw,
                                 	           DataWarehouseP& new_dw) = 0;


        protected:

	 ICELabel* lb;
      };
      
   } // end namespace ICE
} // end namespace Uintah
#endif  // __EQUATION_OF_STATE_H__

// $Log$
// Revision 1.1  2000/10/06 04:02:16  jas
// Move into a separate EOS directory.
//
// Revision 1.4  2000/10/05 04:26:48  guilkey
// Added code for part of the EOS evaluation.
//
// Revision 1.3  2000/10/04 23:41:25  jas
// Get rid of particle stuff.
//
// Revision 1.2  2000/10/04 20:17:51  jas
// Change namespace ICE to ICESpace.
//
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
