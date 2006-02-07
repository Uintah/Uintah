

#ifndef RTRT_HITINFO_H
#define RTRT_HITINFO_H 1

#include <sci_values.h>

namespace rtrt {
  
class Object;

class HitInfo {
public:
  // The distance to the closest intersected object.  This is in the
  // units used by the rays to compute intersections.
  double min_t;

  // The scratchpad data belongs to hit_obj.  It is the responsibility
  // of the object to enforce this.
  char scratchpad[128];

  // The closest object intersected.
  Object* hit_obj;

  // Flag indicating if an object intersection was recorded.
  bool was_hit;
  
  inline HitInfo():min_t(MAXDOUBLE), was_hit(false) {}
  
  // Shadow intersection routines should use shadowhit instead of hit.
  // The idea is to create a HitInfo and assign the distance to the
  // light in min_t.  was_hit will only be set to true if an object's
  // intersection point is closer than min_t.  Regular hit will only
  // test min_t if was_hit is true, thereby making it impossible to
  // check the HitInfo if an occluder was hit.
  inline void shadowHit(Object* obj, double t) {
    if(t>=1.e-6 && t<min_t){
      was_hit=true;
      min_t=t;
      hit_obj=obj;
    }
  }

  // An object trying to become the closest object hit will call this
  // function.
  //
  // Intersection points closer than 1e-4 will be rejected to prevent
  // false self intersecting the same point.
  //
  // If there was no previous hit_obj (was_hit is false), then obj
  // becomes hit_obj and t is recorded as min_t.
  //
  // If there was a previous intersection then min_t is compared to t
  // to determine the closest intersection.
  //
  // If obj becomes hit_obj, then true is returned.  False otherwise.
  inline bool hit(Object* obj, double t) {
    if(t<1.e-6)
      return false;
    if(was_hit){
      if(t<min_t){
	min_t=t;
	hit_obj=obj;
	return true;
      }
    } else {
      was_hit=true;
      min_t=t;
      hit_obj=obj;
      return true;
    }
    return false;
  }
};

} // end namespace rtrt

#endif
