#ifndef __INTERSECTION_GEOMETRY_OBJECT_H__
#define __INTERSECTION_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <Uintah/Grid/Box.h>
#include <vector>
#include <SCICore/Geometry/Point.h>


using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

class IntersectionGeometryObject : public GeometryObject {

 public:
  IntersectionGeometryObject(ProblemSpecP &);
  virtual ~IntersectionGeometryObject();
  
  virtual bool inside(const Point &p) const;
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} // end namespace Components
} // end namespace Uintah

#endif // __INTERSECTION_GEOMETRY_OBJECT_H__
