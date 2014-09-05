#ifndef SOUNDTHREAD_H
#define SOUNDTHREAD_H

#include <Core/Thread/Runnable.h>

#include <dmedia/audiofile.h>
#include <dmedia/audio.h>

#include <vector>

namespace rtrt {

using SCIRun::Runnable;
using std::vector;

class Sound;
class Scene;
class Camera;
class Gui;

class SoundThread : public Runnable {

  friend class Sound;

public:
  SoundThread( const Camera * eyepoint, Scene * scene, Gui * gui );
  virtual ~SoundThread();
  virtual void run();

  void playSound( Sound * sound );

private:
  ALconfig        config_;
  ALport          audioPort_;
  double          samplingRate_;

  vector<Sound*>  soundQueue_;

  const Camera     * eyepoint_;
        Scene      * scene_;

  // These vars are used to communicate with the gui.  Because we don't
  // want to create a circular dependency, lib sound can only call
  // core functions.  So we will ask the gui for the current sound
  // and then update information about the current sound for the gui.
  Gui   * gui_;
  Sound * currentSound_;

  static const int   numChannels_;
};

} // end namespace rtrt

#endif
