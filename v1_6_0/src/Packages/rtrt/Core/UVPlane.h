
#ifndef UVPLANE_H
#define UVPLANE_H 1

#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
  
  using SCIRun::Vector;
  using SCIRun::Point;

class UVPlane : public UVMapping {
    Point cen;
    Vector v1, v2;
public:
    UVPlane(const Point& cen, const Vector& v1, const Vector& v2);
    virtual ~UVPlane();
    virtual void uv(UV& uv, const Point&, const HitInfo& hit);
    virtual void get_frame(const Point &, const HitInfo&,const Vector &,  Vector &v2, Vector &v3)
  {
    v2 = v1;
    v3 = v2;
  }

};

} // end namespace rtrt

#endif

