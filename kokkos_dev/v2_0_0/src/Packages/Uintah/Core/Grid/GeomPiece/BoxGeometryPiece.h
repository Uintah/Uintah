#ifndef __BOX_GEOMETRY_OBJECT_H__
#define __BOX_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Core/Geometry/Point.h>

namespace Uintah {

/**************************************
	
CLASS
   BoxGeometryPiece
	
   Creates a box from the xml input file description.
	
GENERAL INFORMATION
	
   BoxGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   BoxGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a box from the xml input file description.
   Requires two inputs: lower left point and upper right point.  
   There are methods for checking if a point is inside the box
   and also for determining the bounding box for the box (which
   just returns the box itself).
   The input form looks like this:
       <box>
         <min>[0.,0.,0.]</min>
	 <max>[1.,1.,1.]</max>
       </box>
	
	
WARNING
	
****************************************/


      class BoxGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized box.
	 BoxGeometryPiece(ProblemSpecP&);

	 //////////
	 // Construct a box from a min/max point
	 BoxGeometryPiece(const Point& p1, const Point& p2);
	 
	 //////////
	 // Destructor
	 virtual ~BoxGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the box.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 //  Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;
	 
	 //////////
	 //  Returns the volume of the box
	 double volume() const;

	 //////////
	 //  Returns the length pf the smallest side
	 double smallestSide() const;

	 //////////
	 //  Returns the thickness direction (direction
	 //  of smallest side)
	 unsigned int thicknessDirection() const;

      private:
	 Box d_box;
	 
      };
} // End namespace Uintah

#endif // __BOX_GEOMTRY_Piece_H__
