#include "CylinderGeometryObject.h"
#include "GeometryObjectFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Vector.h>


using namespace Uintah::Components;
using namespace SCICore::Geometry;


CylinderGeometryObject::CylinderGeometryObject(ProblemSpecP& ps) {

  Point top,bottom;
  double rad;
  
  ps->require("bottom",bottom);
  ps->require("top",top);
  ps->require("radius",rad);
  
  d_bottom = bottom;
  d_top = top;
  d_radius = rad;
}



CylinderGeometryObject::~CylinderGeometryObject()
{
}

bool CylinderGeometryObject::inside(const Point &p) const
{

  Vector axis = d_top-d_bottom;  
  double height2 = axis.length2();

  Vector tobot = p-d_bottom;

  // pt is the "test" point
  double h = Dot(tobot, axis);
  if(h < 0.0 || h > height2)
    return false; // Above or below the cylinder

  double area = Cross(axis, tobot).length2();
  double d = area/height2;
  if( d > d_radius*d_radius)
    return false;
  return true;

}

Box CylinderGeometryObject::getBoundingBox() const
{
  
  Point lo(d_bottom.x() - d_radius, d_bottom.y() - d_radius,
	   d_bottom.z() - d_radius);

  Point hi(d_top.x() + d_radius, d_top.y() + d_radius,
	   d_top.z() + d_radius);

  return Box(lo,hi);
  

}

// $Log$
// Revision 1.7  2000/04/22 16:51:03  jas
// Put in a skeleton framework for documentation (coccoon comment form).
// Comments still need to be filled in.
//
// Revision 1.6  2000/04/21 22:59:25  jas
// Can create a generalized cylinder (removed the axis aligned constraint).
// Methods for finding bounding box and the inside test are completed.
//
// Revision 1.5  2000/04/20 22:58:13  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.4  2000/04/20 22:37:13  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
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
