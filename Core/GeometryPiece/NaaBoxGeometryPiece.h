#ifndef __NAABOX_GEOMETRY_OBJECT_H__
#define __NAABOX_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Core/Geometry/Vector.h>

namespace SCIRun {
  class Point;
}

namespace Uintah {

/**************************************
	
CLASS
   NaaBoxGeometryPiece
	
   Creates a NON-ACCESS-ALIGNED box from the xml input file description.
	
GENERAL INFORMATION
	
   NaaBoxGeometryPiece.h
	
   J. Davison de St. Germain
   SCI Institute
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS
   Non-Access-Aligned NaaBoxGeometryPiece BoundingBox inside Parallelepiped
	
DESCRIPTION

	
****************************************/

class NaaBoxGeometryPiece : public GeometryPiece {
	 
public:
  //////////
  // Construct a box from four points.
  //
  //       *------------------*
  //      / \                / \
  //    P4___\..............*   \
  //      \   \             .    \
  //       \   P2-----------------*
  //        \  /             .   /
  //         \/               . /
  //         P1---------------P3
  //
  NaaBoxGeometryPiece( const Point& p1, const Point& p2, 
                       const Point& p3, const Point& p4 );

  //////////
  // Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a generalized box.  UPS file should
  // use:
  //      <parallelepiped label = "cube">
  //          <p1>           [1.0, 1.0, 1.0]   </p1>
  //          <p2>           [1.0, 1.5, 1.0]   </p2>
  //          <p3>           [1.5, 1.0, 1.0]   </p3>
  //          <p4>           [1.0, 1.0, 1.5]   </p4>
  //      </parallelepiped>
  NaaBoxGeometryPiece(ProblemSpecP&);

  //////////
  // Destructor
  virtual ~NaaBoxGeometryPiece();

  virtual void outputProblemSpec(ProblemSpecP& ps);

  /// Make a clone
  NaaBoxGeometryPiece* clone();

  //////////
  // Determines whether a point is inside the box.
  virtual bool inside( const Point & pt ) const;
	 
  //////////
  //  Returns the bounding box surrounding the NaaBox
  virtual Box getBoundingBox() const;
	 
private:

  // Called by the different constructors to create the NaaBox
  void init( const Point& p1, const Point& p2, 
             const Point& p3, const Point& p4 );

  Point  p1_, p2_, p3_, p4_;
  Vector p2minusP1_,    p3minusP1_,    p4minusP1_;
  double p2minusP1mag_, p3minusP1mag_, p4minusP1mag_;

  Box boundingBox_;
	 
}; // end class NaaBoxGeometryPiece

} // End namespace Uintah

#endif // __NAABOX_GEOMTRY_Piece_H__
