

#ifndef HITINFO_H
#define HITINFO_H 1

#include <values.h>

namespace rtrt {
  
class Object;

class HitInfo {
public:
    double min_t;
    char scratchpad[128];
    Object* hit_obj;
    bool was_hit;
    inline HitInfo() {
	was_hit=false;
	min_t=MAXDOUBLE;
    }
  inline void shadowHit(Object* obj, double t) {
    if(t>=1.e-4 && t<min_t){
      was_hit=true;
      min_t=t;
      hit_obj=obj;
    }
  }
    inline bool hit(Object* obj, double t) {
	if(t<1.e-4)
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
