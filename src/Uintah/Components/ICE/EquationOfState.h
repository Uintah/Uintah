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
   namespace ICE {
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

	 virtual void addComputesAndRequires(Task* task,
					     const ICEMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const = 0;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to) = 0;

        protected:

	 ICELabel* lb;
      };
      
   } // end namespace ICE
} // end namespace Uintah
#endif  // __EQUATION_OF_STATE_H__

// $Log$
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
