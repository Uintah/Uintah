#ifndef __DIFFERENCE_GEOMETRY_OBJECT_H__
#define __DIFFERENCE_GEOMETRY_OBJECT_H__      

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>

namespace Uintah {

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
	 // Construtor that takes two geometry pieces
	 DifferenceGeometryPiece(GeometryPiece* p1, GeometryPiece* p2);

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
} // End namespace Uintah
      
#endif // __DIFFERENCE_GEOMETRY_Piece_H__
