#ifndef __BOX_GEOMETRY_OBJECT_H__
#define __BOX_GEOMETRY_OBJECT_H__

#include "GeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include <Uintah/Grid/Box.h>


namespace Uintah {
   namespace MPM {
      using SCICore::Geometry::Point;

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
	 // Destructor
	 virtual ~BoxGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the box.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 //  Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;
	 
      private:
	 Box d_box;
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif // __BOX_GEOMTRY_Piece_H__

// $Log$
// Revision 1.3  2000/04/26 06:48:23  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/04/24 21:04:28  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.5  2000/04/22 18:19:10  jas
// Filled in comments.
//
// Revision 1.4  2000/04/22 16:55:11  jas
// Added logging of changes.
//
// Revision 1.3  2000/04/20 18:56:20  sparker
// Updates to MPM
//
// Revision 1.2  2000/04/20 15:09:25  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:07  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
