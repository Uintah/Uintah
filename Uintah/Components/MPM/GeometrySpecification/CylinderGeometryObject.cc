#include "CylinderGeometryObject.h"
#include "GeometryObjectFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Box.h>

using namespace Uintah::Components;


CylinderGeometryObject::CylinderGeometryObject(ProblemSpecP& ps) {

  Point orig;
  double len;
  double rad;
  CylinderGeometryObject::AXIS axis;
  std::string axis_type;

  ps->require("axis",axis_type);
  ps->require("origin",orig);
  ps->require("length",len);
  ps->require("radius",rad);
  
  if (axis_type == "X") axis = CylinderGeometryObject::X;
  if (axis_type == "Y") axis = CylinderGeometryObject::Y;
  if (axis_type == "Z") axis = CylinderGeometryObject::Z;

  d_axis = axis;
  d_origin = orig;
  d_length = len;
  d_radius = rad;


}



CylinderGeometryObject::~CylinderGeometryObject()
{
}

bool CylinderGeometryObject::inside(const Point &p) const
{

  // Do the x axis

  // Do the y axis 

  // Do the z axis

}

Box CylinderGeometryObject::getBoundingBox() const
{


}



// $Log$
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
