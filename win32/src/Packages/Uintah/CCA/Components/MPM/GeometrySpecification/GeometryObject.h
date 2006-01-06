
#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

class GeometryPiece;

using namespace SCIRun;

/**************************************
	
CLASS
   GeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   GeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   GeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

      class MPMMaterial;
      
      class GeometryObject {
	 
      public:
	//////////
	// Insert Documentation Here:
	GeometryObject(MPMMaterial* mpm_matl,GeometryPiece* piece, ProblemSpecP&);

	//////////
	// Insert Documentation Here:
	 ~GeometryObject();

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

	 double getToughnessMin() const {
	    return d_toughnessMin;
	 }

	 double getToughnessMax() const {
	    return d_toughnessMax;
	 }

	 double getToughnessVariation() const {
	    return d_toughnessVariation;
	 }

      private:
	 GeometryPiece* d_piece;
	 IntVector d_resolution;
	 Vector d_initialVel;
	 double d_initialTemperature;

	 double d_toughnessMin;
	 double d_toughnessMax;
	 double d_toughnessVariation;
      };
} // End namespace Uintah
      

#endif // __GEOMETRY_OBJECT_H__


