#ifndef __BOX_GEOMETRY_OBJECT_H__
#define __BOX_GEOMETRY_OBJECT_H__

#include "GeometryObject.h"
#include <SCICore/Geometry/Point.h>
#include <Uintah/Grid/Box.h>

using SCICore::Geometry::Point;
using Uintah::Grid::Box;

namespace Uintah {
namespace Components {

class BoxGeometryObject : public GeometryObject {

 public:

  BoxGeometryObject();
  BoxGeometryObject(Point lower, Point upper);
  virtual ~BoxGeometryObject();

  virtual bool inside(const Point &p) const;
  virtual Box getBoundingBox() const;

  virtual GeometryObject* readParameters(ProblemSpecP &ps);
  
 private:
  Box d_box;

};

} // end namespace Components
} // end namespace Uintah

#endif // __BOX_GEOMTRY_OBJECT_H__

// $Log$
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
