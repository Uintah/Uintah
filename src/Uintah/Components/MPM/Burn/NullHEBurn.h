#ifndef __NULL_HEBURN_H__
#define __NULL_HEBURN_H__

#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>


namespace Uintah {
  namespace MPM {

/**************************************

CLASS
   NullHEBurn
   
   Short description...

GENERAL INFORMATION

   NullHEBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HEBurn_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullHEBurn : public HEBurn { 
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullHEBurn(const NullHEBurn &burn);
      NullHEBurn & operator=(const NullHEBurn &burn);
      
    public:
      // Constructor
      NullHEBurn(ProblemSpecP& ps);
      
      // Destructor
      ~NullHEBurn();

      void checkIfIgnited(const Patch* patch,
			  const MPMMaterial* matl,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

      void computeMassRate(const Patch* patch,
			   const MPMMaterial* matl,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);

      virtual bool getBurns() const;

      // initialize and allocate the burn model data
      void initializeBurnModelData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouseP& new_dw);

      void addCheckIfComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) const;

      void addMassRateComputesAndRequires(Task* task,
					  const MPMMaterial* matl,
					  const Patch* patch,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw) const;

    };
    
  } // end namespace MPM
} // end namespace Uintah

// $Log$
// Revision 1.4  2000/06/19 23:52:14  guilkey
// Added boolean d_burns so that certain stuff only gets done
// if a burn model is present.  Not to worry, the if's on this
// are not inside of inner loops.
//
// Revision 1.3  2000/06/08 16:49:44  guilkey
// Added more stuff to the burn models.  Most infrastructure is now
// in place to change the mass and volume, we just need a little bit of science.
//
// Revision 1.2  2000/06/06 18:04:02  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//

#endif /* __NULL_HEBURN_H__*/

