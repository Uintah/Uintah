#if !defined(linux)
#include <Packages/rtrt/Sound/SoundThread.h>

#include <Packages/rtrt/Sound/Sound.h>

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Gui.h>

#include <Core/Thread/Time.h>
#include <Core/Math/MinMax.h>

#include <dmedia/audiofile.h>
#include <dmedia/audio.h>

#include <iostream>

#include <math.h>
#include <stdio.h>
#include <errno.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

static int portNum = 0;

// We are doing stereo (ie: 2 channels.)  If you change this, a number
// of things will have to be fixed!
const int SoundThread::numChannels_ = 2;

SoundThread::SoundThread( const Camera * eyepoint, Scene * scene,
			  Gui * gui ) :
  eyepoint_( eyepoint ), scene_( scene ), gui_( gui ), sleepTime_(0.0)
{
  ALpv  pv;

  // Make sure that this matches the one in Sound.cc
  int requestedRate = AL_RATE_32000;

  // One tenth (1/10) of a second of sound will be sent down at a time.
  samplingRate_ = requestedRate / 10;

  cout << "samplingRate_: " << samplingRate_ << "\n";

  // Tell the SGI to switch to playing at requestedRate KHz;
  pv.param   = AL_RATE;
  pv.value.i = requestedRate;
  if (alSetParams(AL_DEFAULT_OUTPUT,&pv,1) < 0) {
    cerr << "alSetParams failed: " << alGetErrorString(oserror()) << "\n";
    return;
  }

  /*
   * Create a config which specifies a monophonic port
   * with 1 seconds' worth of space in its queue.
   */
  // Stereo.  If you change this, you need to change the code below
  // that separates the sounds into two channels, and make sure that
  // you sync it with Sound.cc.
  config_ = alNewConfig();
  // Enough space to hold numChannels_ of 1/10 second frames.
  alSetQueueSize( config_, samplingRate_ * 2 );
  alSetSampFmt(   config_, AL_SAMPFMT_TWOSCOMP );
  alSetChannels(  config_, numChannels_ );
  alSetWidth(     config_, AL_SAMPLE_16 );

  char portName[20];
  sprintf( portName, "rtrt sounds %d", portNum );
  portNum++;

  audioPort_ = alOpenPort( portName, "w", config_);
  if( !audioPort_ ) {
    cerr << "Couldn't open port: " << alGetErrorString(oserror()) << "\n";
  }

  soundQueue_ = scene->getSounds();
  cout << "SoundThread: got " << soundQueue_.size() << " sounds!\n";
}

SoundThread::~SoundThread()
{
  if( audioPort_ )
    alClosePort(audioPort_);
  if( config_ )
    alFreeConfig(config_);
}

void
SoundThread::run()
{
  short * quiet = new short[(int)(samplingRate_+1) * numChannels_];
  short * mix   = new short[(int)(samplingRate_+1) * numChannels_];

  // Initialize "quiet" buffer to all -128's.
  for( int cnt = 0; cnt < samplingRate_*numChannels_; cnt++ ) {
    quiet[cnt] = 0; 
  }

  for( int cnt = 0; cnt < soundQueue_.size(); cnt++ )
    {
      soundQueue_[cnt]->load();
    }

  gui_->soundThreadNowActive();

  for(;;)
    {
      int maxFrames = 0;

      if( sleepTime_ )
	{
	  Time::waitFor( sleepTime_ );
	}

      if( soundQueue_.size() > 0 )
	{
	  Point left, right;
	  eyepoint_->get_ears( left, right, 0.3 );

	  const Point & eye = eyepoint_->get_eye();

	  //cout << "eye is " << eye << ", left: " << left 
	  //   << ", right: " << right << "\n";

	  for( int cnt = 0; cnt < samplingRate_*numChannels_; cnt++ )
	    {
	      mix[cnt] = 0;
	    }

	  //vector<Sound*>::iterator iter = soundQueue_.begin();

	  int globalVolume = scene_->soundVolume();

	  for( int cnt = 0; (globalVolume > 0) && (cnt < soundQueue_.size()); 
	       cnt++ )
	    {
	      Sound * sound = soundQueue_[cnt];
	      int     frames;
	      short * buffer;

	      if( !sound->isOn() ) continue;

	      double leftVolume = sound->volume( left );
	      double rightVolume = sound->volume( right );

	      if( leftVolume > 0 || rightVolume > 0 )
		{
		  buffer = sound->getFrames( samplingRate_, frames );

		  //cout << "vol: " <<leftVolume<< ", " << rightVolume << "\n";

		  maxFrames = Max( frames, maxFrames );

		  for( int cnt = 0; cnt < frames; cnt++ )
		    {
		      mix[cnt*2]   += buffer[cnt] * leftVolume  * 
		      	(globalVolume/100.0);
		      mix[cnt*2+1] += buffer[cnt] * rightVolume *
			(globalVolume/100.0);
		    }
		}
	    }
	}

      if( maxFrames == 0 )
	{
	  //cout << "being quiet\n";
	  alWriteFrames(audioPort_, quiet, samplingRate_ );
	  //cout << "done being quiet\n";
	}
      else
	{
	  int fillable = alGetFillable( audioPort_ );

	  //cout << "fillable: " << fillable << "\n";

	  if( maxFrames != samplingRate_ )
	    cout << "writing " << maxFrames << " frames\n";
	  //cout << "making noise " << maxFrames << "\n";
	  alWriteFrames(audioPort_, mix, maxFrames );
	  //cout << "done making noise\n";
	}
    }

} // end run()

void
SoundThread::playSound( Sound * sound )
{
  soundQueue_.push_back( sound );
}
#endif
