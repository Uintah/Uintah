/*
 *  Bench.cc: The Preaty Immersive WorkBench Interface
 *
 *  Written by:
 *   Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECommon/Modules/Salmon/SCIBaWGL.h>
#include <PSECommon/Modules/Salmon/Roe.h>
#include <PSECommon/Modules/Salmon/Salmon.h>

#include <unistd.h>
#include <math.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <string.h>

#include <iostream>
using std::cerr;
using std::endl;

#include <PSECommon/Modules/Salmon/SCIBaWGL.h>

namespace PSECommon {
namespace Modules {

using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;

SCIBaWGLTimer::SCIBaWGLTimer( Roe* r, SCIBaWGL* b )
{ 
  roe = r;
  bawgl = b;
  running = 0;
  exit = 0;

  velocity = 0;
  scaleFrom = 0;

  pinch = BAWGL_PINCH_NONE;
  pinchChange = BAWGL_PINCH_NONE;
  
  stylus = BAWGL_STYLUS_OFF;
  stylusChange = BAWGL_STYLUS_OFF;

  glEye(pinchMatrix);
  glEye(pinchChangeMatrix);
  glEye(stylusMatrix);
  glEye(stylusChangeMatrix);

  stylusID = bawgl->getControllerID(BAWGL_STYLUS);
  pinchID = bawgl->getControllerID(BAWGL_PINCH);

  bawgl->setVirtualViewScaleLimits(0.1, 1000.0);
}

void SCIBaWGLTimer::run( ) //formerly body
{
  while( !exit )
    {
      while( !running ) 
        {         
          if( exit ) break; 
          sginap(0);
        }
  
      while( running )
        {
          if( exit ) break;

	  // force redraw by send message to salmon

	  if(roe->manager->mailbox.numItems() >= 
	     roe->manager->mailbox.size()-1)
	    {
	      cerr << "Redraw event dropped, mailbox full!\n";
	    } 
	  else 
	    {
	      roe->manager->mailbox.send(scinew SalmonMessage(roe->id));
	    }

	  // navigation

	  // pinch throttle controlls for scale and velocity
	  
	  bawgl->getControllerState(pinchID, &pinch, &pinchChange);
	  bawgl->getControllerMatrix(pinchID, BAWGL_LEFT, 
				     pinchMatrix, pinchChangeMatrix, BAWGL_REAL_SPACE);
	  
	  bawgl->getControllerState(stylusID, &stylus, &stylusChange);
	  bawgl->getControllerMatrix(stylusID, BAWGL_LEFT, 
				     stylusMatrix, stylusChangeMatrix, BAWGL_REAL_SPACE);
	  bawgl->getControllerMatrix(stylusID, BAWGL_LEFT, 
				     virtualStylusMatrix, virtualStylusChangeMatrix, BAWGL_VIRTUAL_SPACE);
	 // scale

	  if( BAWGL_PINCH_LEFT_THUMB_LEFT_RING(pinch) )
	    { 
	      if( BAWGL_PINCH_LEFT_THUMB_LEFT_RING(pinchChange) ) 
		{
		  scaleFrom = pinchMatrix[13];
		  scaleOriginal = bawgl->getVirtualViewScale();
		}

	      // change the denominator to change the scale rate
	      
	      bawgl->loadVirtualViewScale(scaleOriginal*exp((scaleFrom - pinchMatrix[13])/6));
	    }

	  // velocity

	  if( BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinch) )
	    {
	      // set the fly through velocity

	      GLfloat tmp[16];
	      GLfloat correctOri[16] = { 0.0, 1.0, 0.0, 0.0,
                                         1.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, -1.0, 0.0,
                                         0.0, 0.0, 0.0, 1.0 };

	      glMatrixMult(tmp, correctOri, stylusMatrix);

	      R2zyx(&azim, &elev, &roll, tmp);

	      if( BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinchChange) )
		{
		  scaleFrom = pinchMatrix[13];
		  azim0 = azim; elev0 = elev;
 		}

	      velocity = (pinchMatrix[13] - scaleFrom) / 13;
	      
	      zyx2R(fly, (azim0-azim)/23, 0, 0);

	      fly[12] = virtualStylusMatrix[0]*velocity;
	      fly[13] = virtualStylusMatrix[1]*velocity;
	      fly[14] = virtualStylusMatrix[2]*velocity;
	      fly[15] = 1.0;

	      bawgl->multVirtualViewMatrix(fly);
	    }

	  if( !BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinch) && 
	      BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinchChange) ) velocity = 0;
	  
	  usleep(20000); // ~50 Hz
        }
    }

  return;
}

void SCIBaWGLTimer::start( void )
{
  running = 1;
}

void SCIBaWGLTimer::stop( void )
{
  running = 0;
}

void SCIBaWGLTimer::quit( void )
{
  exit = 1;
}

SCIBaWGL::SCIBaWGL( char* file )
  : BaWGL(file)
{ 
  shutting_down = false;
}

int SCIBaWGL::start( Roe* r )
{
  shutting_down = false;

  init();

  timer = new SCIBaWGLTimer(r, this);
  timerthread = new Thread(timer, "timer");
  timerthread->detach();

  timer->start();

  return(0);
}

void SCIBaWGL::stop( void )
{
  timer->stop();
  timer->quit();

  shutting_down = true;
}

void SCIBaWGL::shutdown_ok()
{
  if( shutting_down )
    {
      quit();
      shutting_down = false;
    }
}

}}
