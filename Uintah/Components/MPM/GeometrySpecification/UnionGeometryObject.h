#ifndef __UNION_GEOMETRY_OBJECT_H__
#define __UNION_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <vector>
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Point.h>

using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

class UnionGeometryObject : public GeometryObject {

 public:
  UnionGeometryObject(ProblemSpecP &);
  virtual ~UnionGeometryObject();

  virtual bool inside(const Point &p) const;
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} //end namespace Components
} //end namespace Uintah

#endif // __UNION_GEOMETRY_OBJECT_H__
