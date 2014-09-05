
#ifndef PLANEDPY_H
#define PLANEDPY_H 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <X11/Xlib.h>

namespace rtrt {

using SCIRun::Runnable;
using SCIRun::Vector;
using SCIRun::Point;

class PlaneDpy : public DpyBase {
 protected:
    int xres, yres;
    int starty;
    virtual void move(int x, int y);

    virtual void init();
    virtual void display();
    virtual void resize(const int width, const int height);
    virtual void button_released(MouseButton button, const int x, const int y);
    virtual void button_motion(MouseButton button, const int x, const int y);
 public:
    Vector n;
    double d;
    PlaneDpy(const Vector& v, const Point& p);
    PlaneDpy(const Vector& v, const double d);
    virtual ~PlaneDpy();
};

} // end namespace rtrt

#endif

