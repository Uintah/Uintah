
#ifndef TRIGGER_H
#define TRIGGER_H

#include <Core/Geometry/Point.h>

#include <string>
#include <vector>

namespace rtrt {

using std::string;
using std::vector;
using SCIRun::Point;

class Sound;
class SoundThread;
class PPMImage;

// Probably should make image and sound and etc triggers as
// sub-classes, but going to hard code it for now.

class Trigger  {

public:

  // NULL image or sound means trigger doesn't use them.  Trigger can
  // only support a sound or an image, NOT BOTH.  If both are
  // specified, the sound is disregarded.  Distance is distance from
  // any location at which the trigger will activate.
  Trigger( const string  & name,
	   vector<Point> & locations,
	   double          distance,
	   double          delay,
	   PPMImage      * image,
	   bool            showOnGui = false,
	   Sound         * sound = NULL );
  ~Trigger();

  string getName() { return name_; }

  // Force the trigger to activate (usually from the GUI) regardless
  // of whether the eye is near it or not.  Returns true if the
  // trigger remains active (ie: advance() must be called.)  Returns
  // false if trigger is a one time shot (eg: a sound (which will be
  // started)), or if trigger is not ready to be activated yet (ie:
  // timeLeft_ != 0)
  bool activate();

  // Once trigger is triggered, advance should be called until the
  // trigger is done.  Trigger is done when advance returns false;
  bool advance();

  // Returns true if trigger activates (due to being near the eye
  // location) and advance() must be called.  Otherwise returns false
  // (however, trigger may kick off if it is a one time trigger.)
  bool check( const Point & eye );

private:

  string name_;

  vector<Point> locations_;
  double        distance2_;

  Sound    * sound_; // If null, then not part of this trigger.
  PPMImage * image_; 

  // Length of time after this trigger is kicked off before it
  // can be kicked off again.  In SECONDS!  (Also the length of time
  // this trigger takes to run its course.)
  double delay_;

  double timeLeft_; // before trigger can activate again.

  double timeStarted_; // time trigger activated.

  bool   showOnGui_;

};

} // end namespace rtrt

#endif
