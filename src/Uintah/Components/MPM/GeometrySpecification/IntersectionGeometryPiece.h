#ifndef __INTERSECTION_GEOMETRY_OBJECT_H__
#define __INTERSECTION_GEOMETRY_OBJECT_H__      


#include "GeometryPiece.h"
#include <Uintah/Grid/Box.h>
#include <vector>
#include <SCICore/Geometry/Point.h>


namespace Uintah {
   namespace MPM {
      using SCICore::Geometry::Point;


/**************************************
	
CLASS
   IntersectionGeometryPiece
	
   Creates the intersection of geometry pieces from the xml input 
   file description. 

	
GENERAL INFORMATION
	
   IntersectionGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   IntersectionGeometryPiece
	
DESCRIPTION
   Creates a intersection of different geometry pieces from the xml input 
   file description.
   Requires multiple inputs: specify multiple geometry pieces.  
   There are methods for checking if a point is inside the intersection of 
   pieces and also for determining the bounding box for the collection.
   The input form looks like this:
       <intersection>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </intersection>
	
WARNING
	
****************************************/

      class IntersectionGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the intersection of geometry pieces.
	 IntersectionGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Destructor
	 virtual ~IntersectionGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the intersection piece.  
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the intersection piece.
	 virtual Box getBoundingBox() const;
	 
      private:
	 std::vector<GeometryPiece* > child;
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah

#endif // __INTERSECTION_GEOMETRY_PIECE_H__

// $Log$
// Revision 1.2  2000/04/26 06:48:25  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/24 21:04:31  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.5  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.4  2000/04/22 16:55:12  jas
// Added logging of changes.
//
