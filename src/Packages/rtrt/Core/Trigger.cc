

#include <Packages/rtrt/Core/Trigger.h>

#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Sound/Sound.h>
#include <Packages/rtrt/Sound/SoundThread.h>

#include <Core/Thread/Time.h>

#include <iostream>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

PPMImage * blank = NULL;

Trigger::Trigger( const string  & name, 
		  vector<Point> & locations,
		  double          distance,
		  double          delay,
		  PPMImage      * image,
		  bool            showOnGui, /* = false */
		  Sound         * sound,     /* = NULL */
		  bool            fading,    /* = true */
		  Trigger       * next       /* = NULL */ ) :
  name_(name), locations_(locations), distance2_(distance*distance),
  delay_(delay), image_(image), showOnGui_(showOnGui),
  sound_(sound), timeLeft_(0.0), next_(next), fades_(fading),
  blend_(NULL), tex_(NULL), texQuad_(NULL)
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
	      return true;
	    }
	}
    }
  else if( timeLeft_ > 0 )
    {
      double curTime = SCIRun::Time::currentSeconds();
      double elapsedTime = curTime - lastTime_;
      lastTime_ = curTime;
      timeLeft_ -= elapsedTime;

      if( timeLeft_ < 0 )
	{
	  timeLeft_ = 0;
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
      Trigger * dummy;
      lastTime_ = SCIRun::Time::currentSeconds();
      timeLeft_ = delay_;
      return advance( dummy );
    }
  return false;
}

bool
Trigger::advance( Trigger *& next )
{
  // The check() routine decreases "timeLeft_".

  if( sound_ )
    {
      sound_->playNow();
      return false;
    }
  else // image to display
    {
      // If a tex has not been specified, just return.
      if( !texQuad_ || !tex_ ) return false;

      const float timeForFade = 2.0;

      if( fades_ && blend_ )
	{
	  double alpha = 1.0;
	  if( (delay_ - timeLeft_) < timeForFade )
	    { // Image is just starting, fade it in.
	      alpha = (delay_ - timeLeft_) / timeForFade;
	    }

	  if( timeLeft_ < timeForFade )
	    { // Start fading out at timeForFade seconds.
	      alpha = timeLeft_ / timeForFade;
	      blend_->alpha( 1.0 );
	      tex_->reset( GL_FLOAT, &((*blank)(0,0)) );
	      texQuad_->draw(); // Draw blank background
	    }
	  blend_->alpha( alpha );
	  tex_->reset( GL_FLOAT, &((*image_)(0,0)) );
	  texQuad_->draw(); // blend the image over it
	}
      else
	{
	  // This is only for the bottom graphic.
	  tex_->reset( GL_FLOAT, &((*image_)(0,0)) );
	  texQuad_->draw();
	}

      if( timeLeft_ == 0 )
	{
	  next = next_;
	  return false;
	}
      else
	{
	  next = NULL;
	  return true;
	}
    }
}

bool
Trigger::deactivate()
{
  timeLeft_ = Min( timeLeft_, 2.0 );
  if( image_ )
    return true;
  else
    return false;
}

void
Trigger::setDrawableInfo( BasicTexture * tex,
			  ShadedPrim   * texQuad,
			  Blend        * blend /* = NULL */ )
{
  tex_     = tex;
  texQuad_ = texQuad;
  blend_   = blend;
}

