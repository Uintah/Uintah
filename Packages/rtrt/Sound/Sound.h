#ifndef SOUND_H
#define SOUND_H

#include <Core/Geometry/Point.h>

#include <string>
#include <vector>

#include <dmedia/audiofile.h>
#include <dmedia/audio.h>

namespace rtrt {

using SCIRun::Point;

using std::string;
using std::vector;

class Sound {

public:

  // If locations is an empty list, then the sound will be considered
  // to be everywhere.  "Distance" is the distance at which the sound
  // will start becoming audible in units of world space.
  Sound( const string        & filename,
	 const string        & name,
	 const vector<Point> & locations,
	       double          distance = 10,
	       bool            repeat = false  );

  virtual ~Sound();

  // Returns a pointer into the soundBuffer_; increments current sound
  // buffer location.  If there are less frames left in the sound then
  // requested, actualNumframes will hold the number of frames left.
  signed char * getFrames( int numFrames, int & actualNumframes );

  // Actually load the sound.  SOUND WILL NOT PLAY UNTIL THIS IS CALLED!
  void    activate();

  // Continually repeat this sound?
  bool    repeat() { return continuous_; }

  // Volume to play this sound at (percentage from 0 to 100.)
  double volume( const Point & location );

  void currentVolumes( double & right, double & left ) { 
    right = rightVolume_; left = leftVolume_; 
  }

  const string getName() const { return name_; }

private:
  signed char * soundBuffer_;
  int           numFrames_;

  int           bufferLocation_;

  bool          continuous_;

  string        filename_;
  string        name_;

  // A list of the positions that this sound emanates from.
  vector<Point> locations_;
  double        distance2_; // Distance^2 that sound becomes audible.

  double        leftVolume_, rightVolume_;
};

} // end namespace rtrt


#endif
