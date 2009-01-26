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


#ifndef SOUND_H
#define SOUND_H

#include <Core/Geometry/Point.h>

#include <string>
#include <vector>

#if !defined( linux ) && !defined(__APPLE__)
#include <dmedia/audiofile.h>
#include <dmedia/audio.h>
#endif

namespace rtrt {

using SCIRun::Point;

using std::string;
using std::vector;

class GGT;

class Sound {

  friend class GGT;

public:

  // If locations is an empty list, then the sound will be considered
  // to be everywhere.  "Distance" is the distance at which the sound
  // will start becoming audible in units of world space. 
  // If constantVol is > 0, then the sound will play at a constant
  // volume across the entire hearable area.  (constantVol has a vaild
  // range of 0.0 to 1.0)
  Sound( const string        & filename,
	 const string        & name,
	 const vector<Point> & locations,
	       double          distance = 10,
	       bool            repeat = false,
	       double          constantVol = -1 );

  virtual ~Sound();

  // Returns a pointer into the soundBuffer_; increments current sound
  // buffer location.  If there are less frames left in the sound then
  // requested, actualNumframes will hold the number of frames left.
  short * getFrames( int numFrames, int & actualNumframes );

  // Actually load the sound.  SOUND WILL NOT PLAY UNTIL THIS IS CALLED!
  void    load();

  // If sound is on, it will play if you are near it.  If not, no sound.
  void    playNow() { on_ = true; playNow_ = true; }
  bool    isOn() { return on_; }

  // Continually repeat this sound?
  bool    repeat() { return continuous_; }

  // Volume to play this sound at when heard from "location".
  // (percentage from 0 to 100.)
  double volume( const Point & location );

  void currentVolumes( double & right, double & left ) { 
    right = rightVolume_; left = leftVolume_; 
  }

  const string getName() const { return name_; }

private:
  // loaded_ is used so that when a sound is pio'ed back in, when the
  // SoundThread calls activate, the sound will not load itself again.
  bool          loaded_;

  short       * soundBuffer_;
  int           numFrames_;

  int           bufferLocation_;

  bool          continuous_;
  double        constantVol_;

  bool          on_;
  bool          playNow_;

  string        filename_;
  string        name_;

  // A list of the positions that this sound emanates from.
  vector<Point> locations_;
  double        distance2_; // Distance^2 that sound becomes audible.

  double        leftVolume_, rightVolume_;
};

} // end namespace rtrt


#endif
