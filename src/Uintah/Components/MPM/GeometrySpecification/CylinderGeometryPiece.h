#ifndef __CYLINDER_GEOMETRY_OBJECT_H__
#define __CYLINDER_GEOMETRY_OBJECT_H__

#include "GeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include <Uintah/Grid/Box.h>


using SCICore::Geometry::Point;
using Uintah::Grid::Box;

namespace Uintah {
namespace Components {

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

} // end namespace Uintah
} // end namespace Components

#endif // __CYLINDER_GEOMTRY_Piece_H__

// $Log$
// Revision 1.2  2000/04/24 21:04:28  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.7  2000/04/22 18:19:10  jas
// Filled in comments.
//
// Revision 1.6  2000/04/22 16:51:04  jas
// Put in a skeleton framework for documentation (coccoon comment form).
// Comments still need to be filled in.
//
// Revision 1.5  2000/04/21 22:59:25  jas
// Can create a generalized cylinder (removed the axis aligned constraint).
// Methods for finding bounding box and the inside test are completed.
//
// Revision 1.4  2000/04/20 22:58:14  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.3  2000/04/20 22:37:13  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
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
