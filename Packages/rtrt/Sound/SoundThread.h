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

class SoundThread : public Runnable {

public:
  SoundThread( const Camera * eyepoint, Scene * scene );
  virtual ~SoundThread();
  virtual void run();

  void playSound( Sound * sound );

private:
  ALconfig        config_;
  ALport          audioPort_;
  double          samplingRate_;

  vector<Sound*>  soundQueue_;

  const Camera  * eyepoint_;
        Scene   * scene_;
};

} // end namespace rtrt

#endif
