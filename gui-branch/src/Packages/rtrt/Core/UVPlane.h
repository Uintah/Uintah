
#ifndef UVPLANE_H
#define UVPLANE_H 1

#include "UVMapping.h"
#include "Point.h"
#include "Vector.h"

namespace rtrt {

class UVPlane : public UVMapping {
    Point cen;
    Vector v1, v2;
public:
    UVPlane(const Point& cen, const Vector& v1, const Vector& v2);
    virtual ~UVPlane();
    virtual void uv(UV& uv, const Point&, const HitInfo& hit);
};

} // end namespace rtrt

#endif

