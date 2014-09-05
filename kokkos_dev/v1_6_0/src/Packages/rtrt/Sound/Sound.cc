
#include <Packages/rtrt/Sound/Sound.h>
#include <Packages/rtrt/Sound/SoundThread.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

using namespace rtrt;
using namespace std;

Sound::Sound( const string        & filename, 
	      const string        & name, 
	      const vector<Point> & locations,
	            double          distance,
	            bool            repeat /* = false */,
	            double          constantVol /* = -1 */ ) :
  filename_(filename), name_(name), continuous_(repeat), locations_(locations),
  distance2_(distance*distance), bufferLocation_(0), numFrames_(0),
  constantVol_(constantVol)
{
  if( constantVol_ > 1.0 )
    constantVol_ = 1.0;
}

Sound::~Sound()
{
  cout << "~sound: deleteing soundBuffer_\n";
  delete[] soundBuffer_;
}

signed char *
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

  signed char * loc = &(soundBuffer_[ bufferLocation_ ]);

  if( actualNumframes < numFramesRequested ) // reset to beginning
    {
      //cout << "reseting sound\n";
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

  double vol = 0;
  if( minDist2 < distance2_ )
    {
      if( constantVol_ > 0 )
	return constantVol_;

      // vol = (distance2_ - minDist2) / distance2_;
      vol = 1.0 - (sqrt(minDist2) / sqrt(distance2_));
    }

  static bool right = false;
  if( right )
    rightVolume_ = vol;
  else
    leftVolume_ = vol;
  right = !right;

  return vol;
}

void
Sound::activate()
{
  AFfilehandle file = afOpenFile(filename_.c_str(),"r",0);

  cout << "Loading sound file: " << filename_ << "\n";

  // Make sure these match the ones in SoundThread.cc
  //   audiofile (af) virtual file routines are not implemented in the freeware
  //   version of audiofile that we are forced to use because SGI does not
  //   release a 64 bit version of libaudiofile.
  // ALL sound files therefore need to be pre-converted into 16 bit 
  //   integer, 32000 Hz, 1 Channel files.
  // afSetVirtualSampleFormat(file, AF_DEFAULT_TRACK, AF_SAMPFMT_FLOAT, 32);
  // afSetVirtualRate(file, AF_DEFAULT_TRACK, AL_RATE_32000 );
  // 
  // the following line could be wrong even if af is fixed due to 
  // new use of single channel audio files:
  // afSetVirtualChannels(file, AF_DEFAULT_TRACK, SoundThread::numChannels_ );

  // We don't use this...
  int fileChannels = afGetChannels( file, AF_DEFAULT_TRACK );
  cout << "file has num channels: " << fileChannels << "\n";
  if( fileChannels != 1 )
    {
      cout << "Bad audio file.  Must have only one channel!\n";
      numFrames_ = 0; // <- sound won't play
      return;
    }

  numFrames_ = afGetFrameCount( file, AF_DEFAULT_TRACK );
                                                 
  unsigned char * tempBuffer = new unsigned char[ numFrames_ ];
  soundBuffer_ = new signed char[ numFrames_ ];

  int frames = afReadFrames( file,AF_DEFAULT_TRACK, tempBuffer, 
			     numFrames_ );

  float max = -9999;
  float min = 9999;

  for( int cnt = 0; cnt < numFrames_; cnt++ )
    {
      int val = tempBuffer[cnt];
      if( val > max )
	max = val;
      if( val < min )
	min = val;
    }
  cout << "min is " << min << " and max is " << max << "\n";

  float min2 = 99999; float max2 = -99999;
  for( int cnt = 0; cnt < numFrames_; cnt++ )
    {
      // This may not be true:
      //
      // Divide by X means we can play X sounds simultaneously at full
      // volume without going out of range.
      int val = tempBuffer[cnt] - 127;

      soundBuffer_[cnt] = val;

      //cout << cnt << ": " << (int)(soundBuffer_[cnt]) << "\n";
      if( soundBuffer_[cnt] > max2 )
	max2 = soundBuffer_[cnt];
      if( soundBuffer_[cnt] < min2 )
	min2 = soundBuffer_[cnt];
    }
  cout << "MIN IS " << min2 << " AND MAX IS " << max2 << "\n";

  delete [] tempBuffer;

  cout << "read in " << frames << " of a possible " 
       << numFrames_ << " frames\n";

  afCloseFile( file );
}
