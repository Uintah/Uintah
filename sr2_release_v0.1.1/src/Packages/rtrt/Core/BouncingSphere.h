
#ifndef BOUNCINGSPHERE_H
#define BOUNCINGSPHERE_H 1

#include <Packages/rtrt/Core/Sphere.h>

namespace rtrt {

class BouncingSphere : public Sphere {
    Point ocen;
    Vector motion;
public:
    BouncingSphere(Material* matl, const Point& cen, double radius,
		   const Vector& motion);
    virtual ~BouncingSphere();
    virtual void animate(double t, bool& changed);
    virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif
