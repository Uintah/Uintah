
#ifndef UVMAPPING_H
#define UVMAPPING_H 1



#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/vec.h>
#include <Packages/rtrt/Core/HitInfo.h>

namespace SCIRun {
  class Point;
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Vector;

class HitInfo;
class UV;

class UVMapping {
public:
  UVMapping();
  virtual ~UVMapping();
  virtual void uv(UV& uv, const Point&, const HitInfo& hit)=0;
  virtual void get_frame(const Point &, const HitInfo&,const Vector &norm,  Vector &v2, Vector &v3);
};

} // end namespace rtrt

#endif
