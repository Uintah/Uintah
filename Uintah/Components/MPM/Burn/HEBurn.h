#ifndef __HEBURN_H__
#define __HEBURN_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/MPMLabel.h>

#include <math.h>


namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class Patch;
   class VarLabel;
   class Task;
   namespace MPM {
     class MPMMaterial;

/**************************************

CLASS
   HEBurn 
   
   Short description...

GENERAL INFORMATION

   HEBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HEBurn_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

      class HEBurn {
      public:
         // Constructor
	 HEBurn();

	 // Basic burn methods

	  bool isBurnable();
          virtual void checkIfIgnited(const Patch* patch,
				      const MPMMaterial* matl,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) = 0;

          virtual void computeMassRate(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) = 0;


         //////////
         // Create space in data warehouse for burn model data

          virtual bool getBurns() const = 0;

          virtual void initializeBurnModelData(const Patch* patch,
                                               const MPMMaterial* matl,
                                               DataWarehouseP& new_dw) = 0;

          virtual void addCheckIfComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw) const = 0;

          virtual void addMassRateComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw) const = 0;

       protected:
	  bool d_burnable;
	  MPMLabel* lb;
    
      };
      
      
   } // end namespace MPM
} // end namespace Uintah
   
// $Log$
// Revision 1.6  2000/07/05 23:43:31  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.5  2000/06/19 23:52:14  guilkey
// Added boolean d_burns so that certain stuff only gets done
// if a burn model is present.  Not to worry, the if's on this
// are not inside of inner loops.
//
// Revision 1.4  2000/06/17 07:06:35  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.3  2000/06/08 16:49:44  guilkey
// Added more stuff to the burn models.  Most infrastructure is now
// in place to change the mass and volume, we just need a little bit of science.
//
// Revision 1.2  2000/06/06 18:04:01  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.1  2000/06/02 22:48:25  jas
// Added infrastructure for Burn models.
//

#endif // __HEBURN_H__

