#ifndef __TRI_GEOMETRY_OBJECT_H__
#define __TRI_GEOMETRY_OBJECT_H__

#include "GeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include <Uintah/Grid/Box.h>
#include <string>


namespace Uintah {
   namespace MPM {
      using SCICore::Geometry::Point;

/**************************************
	
CLASS
   TriGeometryPiece
	
   Creates a triangulated surface piece from the xml input file description.
	
GENERAL INFORMATION
	
   TriGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   TriGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a triangulated surface piece from the xml input file description.
   Requires one input: file name (convetion use suffix .dat).  
   There are methods for checking if a point is inside the surface
   and also for determining the bounding box for the surface.
   The input form looks like this:
       <tri>
         <file>surface.dat</file>
       </tri>
	
	
WARNING
	
****************************************/

      class TriGeometryPiece : public GeometryPiece {
      public:
	 //////////
	 //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the triangulated surface piece.
	 TriGeometryPiece(ProblemSpecP &);
	 //////////
	 
	 // Destructor
	 virtual ~TriGeometryPiece();
	 
	 //////////
	 // Determins whether a point is inside the triangulated surface.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the triangulated surface.
	 virtual Box getBoundingBox() const;
	 
      private:
	 
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif // __TRI_GEOMETRY_PIECE_H__

// $Log$
// Revision 1.1  2000/06/09 18:38:22  jas
// Moved geometry piece stuff to Grid/ from MPM/GeometryPiece/.
//
// Revision 1.3  2000/04/26 06:48:26  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/04/24 21:04:33  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.6  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.5  2000/04/22 16:55:12  jas
// Added logging of changes.
//
// Revision 1.4  2000/04/22 16:51:04  jas
// Put in a skeleton framework for documentation (coccoon comment form).
// Comments still need to be filled in.
//
// Revision 1.3  2000/04/20 22:37:14  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.2  2000/04/20 15:09:26  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:09  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
