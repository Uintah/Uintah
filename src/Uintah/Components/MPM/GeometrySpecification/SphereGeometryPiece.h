#ifndef __SPHERE_GEOMETRY_OBJECT_H__
#define __SPHERE_GEOMETRY_OBJECT_H__

#include "GeometryPiece.h"
#include <math.h>
#include <SCICore/Geometry/Point.h>

using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {


/**************************************
	
CLASS
   SphereGeometryPiece
	
   Creates a sphere from the xml input file description.
	
GENERAL INFORMATION
	
   SphereGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   SphereGeometryPiece  BoundingBox inside
	
DESCRIPTION
   Creates a sphere from the xml input file description.
   Requires two inputs: origin and a radius.  
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere>
         <origin>[0.,0.,0.]</origin>
	 <radius>2.0</radius>
       </sphere>
	
	
WARNING
	
****************************************/


class SphereGeometryPiece : public GeometryPiece {

 public:
  //////////
  //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a sphere.
  SphereGeometryPiece(ProblemSpecP &);

  //////////
  // Destructor
  virtual ~SphereGeometryPiece();

  //////////
  // Determines whether a point is inside the sphere. 
  virtual bool inside(const Point &p) const;

  //////////
  // Returns the bounding box surrounding the box.
  virtual Box getBoundingBox() const;

 private:
 
  Point d_origin;
  double d_radius;
};

} // end namespace Components
} // end namespace Uintah

#endif // __SPHERE_GEOMETRY_PIECE_H__

// $Log$
// Revision 1.2  2000/04/24 21:04:33  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.6  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.5  2000/04/22 16:51:04  jas
// Put in a skeleton framework for documentation (coccoon comment form).
// Comments still need to be filled in.
//
// Revision 1.4  2000/04/20 22:58:14  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.3  2000/04/20 22:37:14  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.2  2000/04/20 15:09:26  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:08  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
