#ifndef __CONTACT_H__
#define __CONTACT_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>

#include <math.h>

class SimulationStateP;
namespace Uintah {
   using SCICore::Geometry::Vector;
   using SCICore::Math::Min;

   class ProcessorContext;
   class Region;
   class VarLabel;
   namespace MPM {

/**************************************

CLASS
   Contact
   
   Short description...

GENERAL INFORMATION

   Contact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Contact_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

      class Contact {
      public:
         // Constructor
	 Contact(){
	 gMassLabel         = new VarLabel( "g.mass", 
			      NCVariable<double>::getTypeDescription() );
	 gVelocityLabel     = new VarLabel( "g.velocity",
                              NCVariable<Vector>::getTypeDescription() );
	 gVelocityStarLabel = new VarLabel( "g.velocity_star",
                              NCVariable<Vector>::getTypeDescription() );
	 gAccelerationLabel = new VarLabel( "g.acceleration",
                              NCVariable<Vector>::getTypeDescription() );
	 deltLabel          = new VarLabel( "delt",
					    delt_vartype::getTypeDescription() );

	 };

	 // Basic contact methods
	 virtual void exMomInterpolated(const ProcessorContext*,
					const Region* region,
					const DataWarehouseP& old_dw,
					DataWarehouseP& new_dw) = 0;
	 
	 virtual void exMomIntegrated(const ProcessorContext*,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw) = 0;
	 
	 
	 // Auxilliary methods to supply data needed by some of the
	 // advanced contact models
	 virtual void computeSurfaceNormals()
	 {
	    // Null function is the default.  Particular contact
	    // classes will define these functions when needed.
	    return;
	 };
	 virtual void computeTraction()
	 {
	    // Null function is the default.  Particular contact
	    // classes will define these functions when needed.
	    return;
	 };

         // VarLabels common to all contact models go here
         const VarLabel* deltLabel;
         const VarLabel* gMassLabel;
         const VarLabel* gAccelerationLabel;
         const VarLabel* gVelocityLabel;
         const VarLabel* gVelocityStarLabel;
      };
      
      inline bool compare(double num1, double num2)
	 {
	    double EPSILON=1.e-8;
	    
	    return (fabs(num1-num2) <= EPSILON);
	 }
      
      
   } // end namespace MPM
} // end namespace Uintah
   
// $Log$
// Revision 1.9  2000/05/02 06:07:14  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.8  2000/04/27 20:00:25  guilkey
// Finished implementing the SingleVelContact class.  Also created
// FrictionContact class which Scott will be filling in to perform
// frictional type contact.
//
// Revision 1.7  2000/04/26 06:48:20  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/25 22:57:30  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.5  2000/04/12 16:57:27  guilkey
// Converted the SerialMPM.cc to have multimaterial/multivelocity field
// capabilities.  Tried to guard all the functions against breaking the
// compilation, but then who really cares?  It's not like sus has compiled
// for more than 5 minutes in a row for two months.
//
// Revision 1.4  2000/03/21 01:29:41  dav
// working to make MPM stuff compile successfully
//
// Revision 1.3  2000/03/20 23:50:44  dav
// renames SingleVel to SingleVelContact
//
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//

#endif // __CONTACT_H__

