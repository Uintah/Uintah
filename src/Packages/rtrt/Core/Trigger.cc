

#include <Packages/rtrt/Core/Trigger.h>

#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Sound/Sound.h>
#include <Packages/rtrt/Sound/SoundThread.h>

#include <Core/Thread/Time.h>

#include <iostream>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

////////////////////////////////////////////
// OOGL stuff
extern BasicTexture * blendTex;
extern ShadedPrim   * blendTexQuad;
extern Blend        * blend;
PPMImage * blank = NULL;
////////////////////////////////////////////

Trigger::Trigger( const string  & name, 
		  vector<Point> & locations,
		  double          distance,
		  double          delay,
		  PPMImage      * image,
		  bool            showOnGui, /* = false */
		  Sound         * sound /* = NULL */ ) :
  name_(name), locations_(locations), distance2_(distance*distance),
  delay_(delay), image_(image), showOnGui_(showOnGui),
  sound_(sound), timeLeft_(0.0)
{
  if( blank == NULL )
    {
      string path = "/usr/sci/data/Geometry/interface/";
      blank = new PPMImage( path + "baseblend.ppm", true );
    }

  if( sound && image )
    {
      cout << "Trigger with both sound and image.  Sound disregarded.\n";
      sound = NULL;
    }
  if( locations_.size() == 0 )
    {
      cout << "Trigger must have a location.\n";
      image = NULL;
      sound = NULL;
    }
}

Trigger::~Trigger()
{
}


bool
Trigger::check( const Point & eye )
{
  if( timeLeft_ == 0 )
    {
      for( int cnt = 0; cnt < locations_.size(); cnt++ )
	{
	  double dist2 = (eye - locations_[cnt]).length2();
	  if( dist2 < distance2_ )
	    {
	      return activate();
	    }
	}
    }
  else if( timeLeft_ > 0 )
    {
      double elapsedTime = Time::currentSeconds() - timeStarted_;
      timeLeft_ = delay_ - elapsedTime;
      if( timeLeft_ < 0 )
	{
	  timeLeft_ = 0;
	  cout << "Trigger " << name_ << " is ready again\n";
	}
    }
  return false;
}

// Kicks off the trigger.
bool
Trigger::activate()
{
  if( timeLeft_ == 0 ) 
    {
      timeStarted_ = Time::currentSeconds();
      timeLeft_ = delay_;
      return advance();
    }
  cout << "trigger: " << name_ << " is not ready yet ()" << timeLeft_ <<  "\n";
  return false;
}

bool
Trigger::advance()
{
  // The check() routine decreases "timeLeft_".

  if( sound_ )
    {
      cout << "turning on sound: " << name_ << "\n";
      sound_->playNow();
      return false;
    }
  else // image to display
    {
      if( !blend ) return false;  // If we are not in full screen mode.

      double alpha = timeLeft_ / delay_;

      // Don't start fading until only 20% of time is left.
      if( alpha < 0.2 )
	{
	  alpha = 5 * timeLeft_ / delay_; // reset to run from 1.0 to 0.0
	  blend->alpha( 1.0 );
	  blendTex->reset( GL_FLOAT, &((*blank)(0,0)) );
	  blendTexQuad->draw(); // Draw blank background
	}
      else
	{
	  alpha = 1.0;
	}
      blend->alpha( alpha );
      blendTex->reset( GL_FLOAT, &((*image_)(0,0)) );
      blendTexQuad->draw(); // blend the image over it

      return (timeLeft_ != 0);
    }
}


