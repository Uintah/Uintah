#ifndef MOTION_H
#define MOTION_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Scene.h>

#define  NOT_SELECTED 0
#define  CONTINUOUS   1
#define  SEPARATE     2

namespace rtrt {

class Motion {
protected:
  Object *object;      // Selected object
  Vector  u, v;        // Image vectors to determine motion direction
  int     x, y;        // Start screen coordinates
  double  depth;       // Distance to object at start
  double  iyres;       // Y resolution of screen
  Vector  translation; // Cumulative translation since start
  BBox    bbox;        // Bounding box of object at start
  Scene*  scene;       // The scene to be rendered
  int     group_index; // Where in the outside group?
  bool    continuous;  // Continuous spatial subdivision updates?
  double  factor;      // Multiplication factor when moving along line of sight

public:
  int     selected;    // Flag

  Motion          ();
  Motion          (Object *object, int x, int y, double iyres, 
	           const Vector& u, const Vector& v, double depth);
  virtual ~Motion ();
  void set        (Object *object, int x, int y, double iyres, 
	           const Vector& u, const Vector& v, double depth);
  void set_continuous (bool continuous);
  void unset      ();
  void set_scene  (Scene *scene);
  void update     (double nx, double ny);
  void zoom       (double nx, double ny);
  inline Object* get_animated_object () const {
    return object;
  }

  // Timing stuff:

  void   write_stats ();
  void   reset_timer ();
  long   total_updates;
  double total_time;
};

} // end namespace rtrt

#endif
