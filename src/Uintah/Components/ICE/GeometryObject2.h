
#ifndef __GEOMETRY_OBJECT2_H__
#define __GEOMETRY_OBJECT2_H__

#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   class GeometryPiece;
   namespace ICESpace {
      using namespace SCICore::Geometry;
/**************************************
	
CLASS
   GeometryObject2
	
   Short description...
	
GENERAL INFORMATION
	
   GeometryObject2.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS
   GeometryObject2
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

      class ICEMaterial;
      
      class GeometryObject2 {
	 
      public:
	//////////
	// Insert Documentation Here:
	GeometryObject2(ICEMaterial* mpm_matl,GeometryPiece* piece,
		       ProblemSpecP&);

	//////////
	// Insert Documentation Here:
	 ~GeometryObject2();

         //////////
         // Insert Documentation Here:
         IntVector getNumParticlesPerCell();

	 //////////
	 // Insert Documentation Here:
	 GeometryPiece* getPiece() const {
	    return d_piece;
	 }

	 Vector getInitialVelocity() const {
	    return d_initialVel;
	 }

	 double getInitialTemperature() const {
	    return d_initialTemperature;
	 }

      private:
	 GeometryPiece* d_piece;
	 IntVector d_resolution;
	 Vector d_initialVel;
	 double d_initialTemperature;
      };
      
   } // end namespace ICESpace
} // end namespace Uintah

#endif // __GEOMETRY_OBJECT2_H__

// $Log$
// Revision 1.2  2000/11/23 00:45:45  guilkey
// Finished changing the way initialization of the problem was done to allow
// for different regions of the domain to be easily initialized with different
// materials and/or initial values.
//
// Revision 1.1  2000/11/22 01:28:05  guilkey
// Changed the way initial conditions are set.  GeometryObjects are created
// to fill the volume of the domain.  Each object has appropriate initial
// conditions associated with it.  ICEMaterial now has an initializeCells
// method, which for now just does what was previously done with the
// initial condition stuct d_ic.  This will be extended to allow regions of
// the domain to be initialized with different materials.  Sorry for the
// lame GeometryObject2, this could be changed to ICEGeometryObject or
// something.
//
