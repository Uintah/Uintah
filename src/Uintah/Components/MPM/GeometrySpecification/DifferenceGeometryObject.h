#ifndef __DIFFERENCE_GEOMETRY_OBJECT_H__
#define __DIFFERENCE_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Point.h>


using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

class DifferenceGeometryObject : public GeometryObject {

 public:
  DifferenceGeometryObject();
  virtual ~DifferenceGeometryObject();
  DifferenceGeometryObject(const DifferenceGeometryObject& copy);

  virtual void add(const GeometryObject* go);
  virtual bool inside(const Point &p) const;
  virtual Box getBoundingBox() const;

 private:
  GeometryObject* left;
  GeometryObject* right;


};

} // end namespace Components
} // end namespace Uintah

#endif // __DIFFERENCE_GEOMETRY_OBJECT_H__
