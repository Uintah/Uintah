

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

const double Trigger::HighTriggerPriority   = 100.0; // <- Gui uses.
const double Trigger::MediumTriggerPriority =  50.0; // <- Normal objects.
const double Trigger::LowTriggerPriority    =   0.0; // <- Background objects.

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
  sound_(sound), image_(image), tex_(NULL), texQuad_(NULL), blend_(NULL),
  delay_(delay), timeLeft_(0.0), showOnGui_(showOnGui), fades_(fading),
  next_(next), priority_(LowTriggerPriority), basePriority_(LowTriggerPriority)
{
  if( blank == NULL )
    {
      string path = "/usr/sci/data/Geometry/interface/backgrounds/";
      blank = new PPMImage( path + "text_area.ppm", true );
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


void
Trigger::setPriority( double priority )
{
  priority_ = priority;
}

bool
Trigger::check( const Point & eye )
{
  // Since sounds are not "advance()"d, we use "check()" to deteremine
  // when their delay is over.
  if( sound_ && timeLeft_ > 0 )
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


  if( timeLeft_ == 0 )
    {
      for( unsigned int cnt = 0; cnt < locations_.size(); cnt++ )
	{
	  double dist2 = (eye - locations_[cnt]).length2();
	  if( dist2 < distance2_ )
	    {
	      return true;
	    }
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
  // Since some triggers may not be in the check list (nexts of
  // another trigger), we need to advance their time here.
  if( !sound_ && timeLeft_ > 0 )
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

  if( sound_ )
    {
      sound_->playNow();
      return false;
    }
#if defined(HAVE_OOGL)
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
	  priority_ = basePriority_;
	  next = next_;
	  return false;
	}
      else
	{
	  next = NULL;
	  return true;
	}
    }
#else
  return false;
#endif
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

