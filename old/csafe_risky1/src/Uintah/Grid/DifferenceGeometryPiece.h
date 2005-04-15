#ifndef __DIFFERENCE_GEOMETRY_OBJECT_H__
#define __DIFFERENCE_GEOMETRY_OBJECT_H__      


#include "GeometryPiece.h"
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {
   namespace MPM {
      using SCICore::Geometry::Point;

/**************************************
	
CLASS
   DifferenceGeometryPiece
	
   Creates the difference between two geometry Pieces from the xml input 
   file description. 


GENERAL INFORMATION
	
   DifferenceGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   DifferenceGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates the difference between two  geometry Pieces from the xml input 
   file description.
   Requires tow inputs: specify two geometry Pieces. The order is important.
   There are methods for checking if a point is inside the difference of 
   Pieces and also for determining the bounding box for the collection.
   The input form looks like this:
       <difference>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </difference>

	
WARNING
	
****************************************/

      class DifferenceGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the union of geometry Pieces.
	 DifferenceGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Destructor
	 virtual ~DifferenceGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the union Piece.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the union Piece.
	 virtual Box getBoundingBox() const;
	 
      private:
	 GeometryPiece* left;
	 GeometryPiece* right;
	 
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif // __DIFFERENCE_GEOMETRY_Piece_H__

// $Log$
// Revision 1.1  2000/06/09 18:38:21  jas
// Moved geometry piece stuff to Grid/ from MPM/GeometryPiece/.
//
// Revision 1.2  2000/04/26 06:48:23  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/24 21:04:29  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.5  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.4  2000/04/22 16:55:12  jas
// Added logging of changes.
//
