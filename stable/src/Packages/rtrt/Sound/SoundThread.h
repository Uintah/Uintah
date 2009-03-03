/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef SOUNDTHREAD_H
#define SOUNDTHREAD_H

#include <Core/Thread/Runnable.h>

#if !defined( linux ) && !defined(__APPLE__)
#include <dmedia/audiofile.h>
#include <dmedia/audio.h>
#endif

#include <vector>

namespace rtrt {

using SCIRun::Runnable;
using std::vector;

class Sound;
class Scene;
class Camera;
class GGT;

class SoundThread : public Runnable {

  friend class Sound;

public:
  SoundThread( const Camera * eyepoint, Scene * scene, GGT * gui );
  virtual ~SoundThread();
  virtual void run();

  void playSound( Sound * sound );

  //// Used to shut off the thread from using up CPU resources.
  // Put the sound thread to sleep for N seconds.
  void goToSleep( double seconds ) { sleepTime_ = seconds; }

private:
#if !defined(linux) && !defined(__APPLE__)
  ALconfig        config_;
  ALport          audioPort_;
#endif
  double          samplingRate_;

  vector<Sound*>  soundQueue_;

  double             sleepTime_;

  const Camera     * eyepoint_;
        Scene      * scene_;

  // These vars are used to communicate with the gui.  Because we don't
  // want to create a circular dependency, lib sound can only call
  // core functions.  So we will ask the gui for the current sound
  // and then update information about the current sound for the gui.
  GGT   * gui_;
  Sound * currentSound_;

  static const int   numChannels_;
};

} // end namespace rtrt

#endif
