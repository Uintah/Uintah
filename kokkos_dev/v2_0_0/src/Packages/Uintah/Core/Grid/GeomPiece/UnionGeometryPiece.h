#ifndef __UNION_GEOMETRY_OBJECT_H__
#define __UNION_GEOMETRY_OBJECT_H__      

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
/**************************************
	
CLASS
   UnionGeometryPiece
	
   Creates a collection of geometry pieces from the xml input 
   file description. 
	
GENERAL INFORMATION
	
   UnionGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   UnionGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a union of different geometry pieces from the xml input 
   file description.
   Requires multiple inputs: specify multiple geometry pieces.  
   There are methods for checking if a point is inside the union of 
   pieces and also for determining the bounding box for the collection.
   The input form looks like this:
       <union>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </union>
	
	
WARNING
	
****************************************/

      class UnionGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the intersection of geometry pieces.
	 UnionGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Constructor that takes an array of children. It copies the array,
	 // and assume ownership of the children.
	 UnionGeometryPiece(const std::vector<GeometryPiece*>& children);
	 
	 //////////
	 // Destructor
	 virtual ~UnionGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the intersection piece.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the union piece.
	 virtual Box getBoundingBox() const;
	 
      private:
	 std::vector<GeometryPiece* > child;
	 
      };
} // End namespace Uintah
      

#endif // __UNION_GEOMETRY_PIECE_H__

