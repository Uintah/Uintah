// Friction.h

#ifndef __FRICTION_H__
#define __FRICTION_H__

#include <Uintah/Components/MPM/Contact/Contact.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>

class SimulationStateP;

namespace Uintah {
   class VarLabel;
   namespace MPM {

/**************************************

CLASS
   FrictionContact
   
   Short description...

GENERAL INFORMATION

   FrictionContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model_Friction

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply Coulombic frictional contact.
  
WARNING
  
****************************************/

      class FrictionContact : public Contact {
      private:
	 
	 // Prevent copying of this class
	 // copy constructor
	 FrictionContact(const FrictionContact &con);
	 FrictionContact& operator=(const FrictionContact &con);
	 
	 SimulationStateP d_sharedState;

         // Coefficient of friction
         double d_mu;

         // VarLabels specific to Frictional contact
	 const VarLabel* gNormTractionLabel;
	 const VarLabel* gSurfNormLabel; 
	 const VarLabel* gStressLabel; 
	 const VarLabel* pStressLabel; 
	 const VarLabel* pXLabel; 
	 
      public:
	 // Constructor
	 FrictionContact(ProblemSpecP& ps, SimulationStateP& d_sS);
	 
	 // Destructor
	 virtual ~FrictionContact();
	 
	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorContext*,
					const Region* region,
					const DataWarehouseP& old_dw,
					DataWarehouseP& new_dw);
	 
	 virtual void exMomIntegrated(const ProcessorContext*,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif /* __FRICTION_H__ */

// $Log$
// Revision 1.4  2000/05/05 22:37:27  bard
// Added frictional contact logic.  Compiles but doesn't yet work.
//
// Revision 1.3  2000/05/05 02:24:35  guilkey
// Added more stuff to FrictionContact, most of which is currently
// commented out until a compilation issue is resolved.
//
// Revision 1.2  2000/04/27 21:28:58  jas
// Contact is now created using a factory.
//
// Revision 1.1  2000/04/27 20:00:26  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
