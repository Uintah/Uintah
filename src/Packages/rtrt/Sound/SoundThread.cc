#include <Packages/rtrt/Sound/SoundThread.h>

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>

#include <Core/Math/MinMax.h>

#include <dmedia/audiofile.h>
#include <dmedia/audio.h>

#include <iostream>

#include <math.h>
#include <stdio.h>
#include <errno.h>

#include <Packages/rtrt/Sound/Sound.h> //just for testing

using namespace rtrt;
using namespace std;
using namespace SCIRun;

static int portNum = 0;

SoundThread::SoundThread( const Camera * eyepoint, Scene * scene ) :
  eyepoint_( eyepoint ), scene_( scene )
{
  ALpv  pv;

  int   requestedRate = AL_RATE_44100;

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
  config_ = alNewConfig();
  alSetQueueSize( config_, samplingRate_*2 );
  alSetSampFmt( config_, AL_SAMPFMT_FLOAT );
  alSetChannels( config_, 1 );

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
  float * quiet = new float[ (int)(samplingRate_+1) ];
  float * mix   = new float[ (int)(samplingRate_+1) ];

  // Initialize "quiet" buffer to all 0's.
  for( int cnt = 0; cnt < samplingRate_; cnt++ ) { quiet[cnt] = 0; }

  for( int cnt = 0; cnt < soundQueue_.size(); cnt++ )
    {
      soundQueue_[cnt]->activate();
    }

  for(;;)
    {
      int maxFrames = 0;

      if( soundQueue_.size() > 0 )
	{
	  const Point & eye = eyepoint_->get_eye();

	  for( int cnt = 0; cnt < samplingRate_; cnt++ )
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
	      float * buffer;
	      double  volume = sound->volume( eye );

	      //cout << cnt << ": volume is " << volume << "\n";

	      if( volume > 0 )
		{
		  buffer = sound->getFrames( samplingRate_, frames );

		  maxFrames = Max( frames, maxFrames );

		  if( frames < samplingRate_ && !sound->repeat() ) 
		    {
		      cout << "removing sound\n";
		      //iter = soundQueue_.erase( iter ); // ran out of sound
		      soundQueue_.pop_back(); // ran out of sound
		    }
		  for( int cnt = 0; cnt < frames; cnt++ )
		    {
		      mix[cnt] += buffer[cnt] * volume * (globalVolume/100.0);
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
