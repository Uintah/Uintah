#ifndef __UNION_GEOMETRY_OBJECT_H__
#define __UNION_GEOMETRY_OBJECT_H__      

#include <Core/GeometryPiece/GeometryPiece.h>

#include <sgi_stl_warnings_off.h>
#include   <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/GeometryPiece/uintahshare.h>
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

      class UINTAHSHARE UnionGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the intersection of geometry pieces.
	 UnionGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Constructor that takes an array of children. It copies the array,
	 // and assume ownership of the children.
	 UnionGeometryPiece(const std::vector<GeometryPieceP>& children);

	 /// Assignment operator
	 UnionGeometryPiece& operator=(const UnionGeometryPiece& );

	 //// Make a clone
	 GeometryPieceP clone() const;

	 //////////
	 // Destructor
         virtual ~UnionGeometryPiece() {}
	 
         static const string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

	 //////////
	 // Determines whether a point is inside the intersection piece.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the union piece.
	 virtual Box getBoundingBox() const;
	 
      private:
         virtual void outputHelper( ProblemSpecP & ps ) const;

	 std::vector<GeometryPieceP> child_;
	 
      };
} // End namespace Uintah
      

#endif // __UNION_GEOMETRY_PIECE_H__

