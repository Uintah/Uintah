#include <Core/Thread/Mutex.h>

namespace rtrt {

  class BallData;
  class Camera;

  struct DpyPrivate {

    SCIRun::Mutex * waitDisplay;

    int   xres,yres;

    bool  exposed;
    bool  animate;

    int   showing_scene;

    int   maxdepth;

    bool  followPath;

    float base_threshold;
    float full_threshold;

    int   left;
    int   up;

    bool  stereo;

    BallData * ball;

    double last_time;

    double FrameRate;

    int doing_frameless;

  };

} // end namespace rtrt
  
