
#ifndef TRIGGER_H
#define TRIGGER_H

#include <Packages/rtrt/Core/Image.h>

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

  static const double HighTriggerPriority;    // <- Gui uses.
  static const double MediumTriggerPriority;  // <- Normal objects.
  static const double LowTriggerPriority;     // <- Background objects.

  // NULL image or sound means trigger doesn't use them.  Trigger can
  // only support a sound or an image, NOT BOTH.  If both are
  // specified, the sound is disregarded.  Distance is distance from
  // any location at which the trigger will activate.
  // If "fading" then the image fades in and out.
  Trigger( const string  & name,
	   vector<Point> & locations,
	   double          distance,
	   double          delay,
	   PPMImage      * image,
	   bool            showOnGui = false,
	   Sound         * sound = NULL,
	   bool            fading = true,
	   Trigger       * next = NULL );
  ~Trigger();

  string getName() { return name_; }

  // Force the trigger to activate (usually from the GUI) regardless
  // of whether the eye is near it or not.  Returns true if the
  // trigger remains active (ie: advance() must be called.)  Returns
  // false if trigger is a one time shot (eg: a sound (which will be
  // started)), or if trigger is not ready to be activated yet (ie:
  // timeLeft_ != 0)
  bool activate();

  // Tell the trigger to stop as soon as possible. Image triggers will
  // fade out in 2 seconds.  Returns true if the trigger needs to be
  // advance()'d.
  bool deactivate();

  // Once trigger is triggered, advance should be called until the
  // trigger is done.  Trigger is done when advance returns false;
  // When trigger is done, if it has a next trigger, it will return
  // that trigger.
  bool advance( Trigger *& next );

  // Returns true if trigger activates (due to being near the eye
  // location) and advance() must be called.  Otherwise returns false
  // (however, trigger may kick off if it is a one time trigger.)
  bool check( const Point & eye );

  bool isSoundTrigger() { return sound_; }

  void      setNext( Trigger * next ) { next_ = next; }
  Trigger * getNext() { return next_; }

  void setDelay( double delay ) { delay_ = delay; }

  void setDrawableInfo( BasicTexture * tex,
			ShadedPrim   * texQuad,
			Blend        * blend = NULL );

  void   setPriority( double priority );
  double getPriority() const { return priority_; }
  // setBasePriority should be called once to establish a base priority
  // for this trigger.  If not called, basePriority is 'LowTriggerPriority'.
  void   setBasePriority( double priority ) { basePriority_ = priority; }

private:

  string name_;

  vector<Point> locations_;
  double        distance2_;

  Sound    * sound_; // If null, then not part of this trigger.
  PPMImage * image_; 

  // Location to draw image.
  BasicTexture * tex_;
  ShadedPrim   * texQuad_;
  Blend        * blend_;

  // Length of time after this trigger is kicked off before it
  // can be kicked off again.  In SECONDS!  (Also the length of time
  // this trigger takes to run its course.)
  double delay_;

  double timeLeft_; // before trigger can activate again.

  double lastTime_; // last time trigger was checked.

  bool   showOnGui_;

  bool   fades_;

  Trigger * next_; // If !NULL, this trigger will trigger another trigger
                   // when done.

  // Priority_ is the current priority.  Base is the original.
  // After a trigger is done, its "priority_" is reset to the original.
  double    priority_;
  double    basePriority_;
};

} // end namespace rtrt

#endif
