#ifndef __CYLINDER_GEOMETRY_OBJECT_H__
#define __CYLINDER_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Core/Geometry/Point.h>

namespace Uintah {

/**************************************
	
CLASS
   CylinderGeometryPiece
	
   Creates a generalized cylinder from the xml input file description.
	
GENERAL INFORMATION
	
   CylinderGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   CylinderGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a generalized cylinder from the xml input file description.
   Requires three inputs: bottom point, top point and a radius.  
   There are methods for checking if a point is inside the cylinder
   and also for determining the bounding box for the cylinder.
   The input form looks like this:
       <cylinder>
         <bottom>[0.,0.,0.]</bottom>
	 <top>[0.,0.,0.]</top>
	 <radius>2.0</radius>
       </cylinder>
	
WARNING
	
****************************************/

      class CylinderGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized cylinder.
	 CylinderGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Destructor
	 virtual ~CylinderGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the cylinder.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;
	 
      private:
	 Point d_bottom;
	 Point d_top;
	 double d_radius;
	 
	 
	 
      };
} // End namespace Uintah
      
#endif // __CYLINDER_GEOMTRY_Piece_H__
