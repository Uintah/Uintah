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


#include <math.h>


namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorContext;
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
          virtual void checkIfIgnited() = 0;
          virtual void computeMassRate() = 0;
          virtual void updatedParticleMassAndVolume() = 0;
	
       protected:
	  bool d_burnable;
    
      };
      
      
   } // end namespace MPM
} // end namespace Uintah
   
// $Log$
// Revision 1.1  2000/06/02 22:48:25  jas
// Added infrastructure for Burn models.
//

#endif // __HEBURN_H__

