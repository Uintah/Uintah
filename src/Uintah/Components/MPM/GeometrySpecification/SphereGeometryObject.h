#ifndef __SPHERE_GEOMETRY_OBJECT_H__
#define __SPHERE_GEOMETRY_OBJECT_H__

#include "GeometryObject.h"
#include <math.h>
#include <SCICore/Geometry/Point.h>

using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

class SphereGeometryObject : public GeometryObject {

 public:

  SphereGeometryObject();
  SphereGeometryObject(const double r, const Point o);
  virtual ~SphereGeometryObject();
 
  virtual void add(SphereGeometryObject* go);

  virtual bool inside(const Point &p) const;
  virtual Box getBoundingBox() const;

 private:
 
  Point d_origin;
  double d_radius;
};

} // end namespace Components
} // end namespace Uintah

#endif // __SPHERE_GEOMETRY_OBJECT_H__

// $Log$
// Revision 1.1  2000/04/19 21:31:08  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//
