
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
  int starty;
  virtual void move(int x, int y);
  
  virtual void init();
  virtual void display();
  virtual void resize(const int width, const int height);
  virtual void key_pressed(unsigned long key);
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
  
public:
  Vector n;
  double d;
  PlaneDpy(const Vector& v, const Point& p,
	   bool active = true, bool use_material = true);
  PlaneDpy(const Vector& v, const double d,
	   bool active = true, bool use_material = true);
  virtual ~PlaneDpy();

  bool active;
  bool use_material;
};

} // end namespace rtrt

#endif

