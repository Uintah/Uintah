
#include <Packages/rtrt/Sound/Sound.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

using namespace rtrt;
using namespace std;

Sound::Sound( const string        & filename, 
	      const vector<Point> & locations,
	            double          distance,
	            bool            repeat /* = false */ ) :
  filename_(filename), continuous_(repeat), locations_(locations),
  distance2_(distance*distance), bufferLocation_(0), numFrames_(0)
{
}

Sound::~Sound()
{
  cout << "~sound: deleteing soundBuffer_\n";
  delete[] soundBuffer_;
}

float *
Sound::getFrames( int numFramesRequested, int & actualNumframes )
{
  if( bufferLocation_ + numFramesRequested >= numFrames_ )
    {
      actualNumframes = numFrames_ - bufferLocation_;
    }
  else
    {
      actualNumframes = numFramesRequested;
    }

  float * loc = &(soundBuffer_[ bufferLocation_ ]);

  if( actualNumframes < numFramesRequested ) // reset to beginning
    {
      cout << "reseting sound\n";
      bufferLocation_ = 0;
    }
  else
    {
      bufferLocation_ += actualNumframes;
    }

  return loc;
}

double
Sound::volume( const Point & location )
{
  int size = locations_.size();
  if( size == 0 )
    return 0;

  double minDist2 = 9999999.9;
  for( int cnt = 0; cnt < size; cnt++ )
    {
      double dist2 = (locations_[cnt] - location).length2();
      if( dist2 < minDist2 )
	minDist2 = dist2;
    }

  double volume = 0;
  if( minDist2 < distance2_ )
    {
      volume = (distance2_ - minDist2) / distance2_;
    }
  return volume;
}

void
Sound::activate()
{
  AFfilehandle file = afOpenFile(filename_.c_str(),"r",0);

  afSetVirtualSampleFormat(file, AF_DEFAULT_TRACK, AF_SAMPFMT_FLOAT, 32);
  afSetVirtualRate(file, AF_DEFAULT_TRACK, 44100.0 );

  int numChannels = afGetChannels( file, AF_DEFAULT_TRACK );
  numFrames_ = afGetFrameCount( file, AF_DEFAULT_TRACK );

  cout << "num channels: " << numChannels << "\n";

  soundBuffer_ = new float[ numFrames_ ];

  int frames = afReadFrames( file,AF_DEFAULT_TRACK, soundBuffer_, 
			     numFrames_ );

  cout << "read in " << frames << " of a possible " 
       << numFrames_ << " frames\n";

  afCloseFile( file );
}
