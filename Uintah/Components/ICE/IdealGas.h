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

	 virtual void addComputesAndRequires(Task* task,
					     const ICEMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;


        protected:

	 ICELabel* lb;
      };
      
   } // end namespace ICE
} // end namespace Uintah
#endif  // __IDEAL_GAS_H__

// $Log$
// Revision 1.1  2000/10/04 23:40:12  jas
// The skeleton framework for an EOS model.  Does nothing.
//

