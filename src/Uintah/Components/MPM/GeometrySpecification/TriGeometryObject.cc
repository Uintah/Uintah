#include "TriGeometryObject.h"
#include "GeometryObjectFactory.h"

using namespace Uintah::Components;

TriGeometryObject::TriGeometryObject()
{
}

TriGeometryObject::TriGeometryObject(std::string file)
{
  // Open file and read in crack data
}


TriGeometryObject::~TriGeometryObject()
{
}

bool TriGeometryObject::inside(const Point &p) const
{

}

Box TriGeometryObject::getBoundingBox() const
{

}

GeometryObject* TriGeometryObject::readParameters(ProblemSpecP &ps)
{
  std::string file;

  ps->require("file",file);
  
  return (new TriGeometryObject(file));
  
}

// $Log$
// Revision 1.2  2000/04/20 15:09:26  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:09  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
