
#ifndef MOUSECALLBACK_H
#define MOUSECALLBACK_H

#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>

namespace rtrt {
  class Object;
  class Ray;
  class HitInfo;

  typedef void (*cbFunc)( Object *obj, const Ray&, const HitInfo& );

/*    typedef struct MouseCBGroup { */
/*      cbFunc mouseDown; */
/*      cbFunc mouseUp; */
/*      cbFunc mouseMotion; */
/*    } MouseCBGroup; */
  
  using namespace std;
  
  class MouseCallBack {
  public:
    
    // assigns a callback to an object
/*      static void assignCB( MouseCBGroup &cbGroup, const Object* object); */
    static void assignCB_MD( cbFunc fp, const Object* object );
    static void assignCB_MU( cbFunc fp, const Object* object );
    static void assignCB_MM( cbFunc fp, const Object* object );
    // retrieves a callback from an object
    static cbFunc getCB_MD( const Object* object );
    static cbFunc getCB_MU( const Object* object );
    static cbFunc getCB_MM( const Object* object );
    // determines whether or not an object has been assigned a callback
    static bool hasCB_MD( const Object* object );
    static bool hasCB_MU( const Object* object );
    static bool hasCB_MM( const Object* object );
  };
}

#endif
