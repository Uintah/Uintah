/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SCIBawGL.cc: The Pretty Immersive WorkBench Interface
 *
 *  Written by:
 *   Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Render/SCIBaWGL.h>
#include <Dataflow/Modules/Render/ViewWindow.h>
#include <Dataflow/Modules/Render/Viewer.h>

#include <unistd.h>
#include <math.h>

#include <GL/gl.h>
#include <sci_glu.h>
#include <string.h>

#include <iostream>
using std::cerr;

#include <Dataflow/Modules/Render/SCIBaWGL.h>

namespace SCIRun {


SCIBaWGLTimer::SCIBaWGLTimer( ViewWindow* r, SCIBaWGL* b )
{ 
  viewwindow = r;
  bawgl = b;
  running = 0;
  exit = 0;

  velocity = 0;
  scaleFrom = 0;

  pinch = BAWGL_PINCH_NONE;
  pinch0 = BAWGL_PINCH_NONE;
  pinchChange = BAWGL_PINCH_NONE;
  
  stylus = BAWGL_STYLUS_OFF;
  stylus0 = BAWGL_STYLUS_OFF;
  stylusChange = BAWGL_STYLUS_OFF;
  
  glEye(realPinchMatrix);
  glEye(realStylusMatrix);
  
  glEye(virtualStylusMatrix);
  glEye(virtualStylusChangeMatrix);

  glEye(surfacePinchMatrix);
  
  glEye(virtualPinchChangeMatrix);

  stylusID = bawgl->getControllerID(BAWGL_STYLUS);
  pinchID = bawgl->getControllerID(BAWGL_PINCH);

  navSpeed = 8;

  bawgl->setVirtualViewScaleLimits(0.01, 10000.0);
  bawgl->loadVirtualViewScale(50.0);
}

void SCIBaWGLTimer::navigate( void )
{
  // navigation

  // pinch throttle controlls for scale and velocity
	 
  // scale

  if( BAWGL_PINCH_LEFT_THUMB_LEFT_PINKY(pinch) )
    {
      if( BAWGL_PINCH_LEFT_THUMB_LEFT_PINKY(pinchChange) ) 
	{
	  bawgl->scaleFrom = scaleFrom = realPinchMatrix[13];
	  scaleOriginal = bawgl->getVirtualViewScale();
	  bawgl->scale = 1;
	}

      // change the denominator to change the scale rate
	      
      bawgl->loadVirtualViewScale(scaleOriginal*exp((scaleFrom - realPinchMatrix[13])/6));
    }
  else
    bawgl->scale = 0;

  // navigation
	  
  if( BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinch) )
    {
      if( BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(pinchChange) )
	{
	  bawgl->navigateFrom = navigateFrom = realPinchMatrix[13];
	  bawgl->navigate = 1;
	}

      bawgl->velocity = velocity = (navigateFrom - realPinchMatrix[13]) / (navSpeed * bawgl->virtualViewScale);

      glEye(fly);
      
      fly[12] = virtualStylusMatrix[4]*velocity;
      fly[13] = virtualStylusMatrix[5]*velocity;
      fly[14] = virtualStylusMatrix[6]*velocity;
      fly[15] = 1.0;
      
      bawgl->multVirtualViewMatrix(fly);
    }
  else
    bawgl->navigate = 0;

  // stylus based orientation
  
  if( stylus == BAWGL_STYLUS_ON )
    {
      bawgl->multVirtualViewMatrix(virtualStylusChangeMatrix);
    }
  
  // navigation #2

  if( BAWGL_PINCH_LEFT_THUMB_LEFT_RING(pinch) )
    {
      GLfloat fly1[16], fly2[16], svel, dazim, droll;
      
      R2zyx(&azim, &elev, &roll, realStylusMatrix);
      
      if( BAWGL_PINCH_LEFT_THUMB_LEFT_RING(pinchChange) )
	{
	  scaleFrom = realPinchMatrix[13];
	  azim0 = azim; elev0 = elev; roll0 = roll;
	}
      
      velocity = (realPinchMatrix[13] - scaleFrom) / (navSpeed * bawgl->virtualViewScale);
      
      bawgl->getInverseVirtualViewMatrix(realStylusMatrix);
      
      glEye(fly);
      glEye(fly1);
      glEye(fly2);
      
      svel = velocity > 0.0 ? 1.0 : (velocity < 0.0 ? -1.0 : 0.0);
      
      /* glAxis2Rot(fly1, &tmp[8], svel*(azim-azim0)/37);
	 glAxis2Rot(fly2, &tmp[0], svel*(roll-roll0)/37);
	 glMatrixMult(fly, fly1, fly2); */
      
      dazim = svel*(azim0-azim)/37;
      droll = svel*(roll0-roll)/37;
      
      zyx2R(fly, dazim, 0, droll);
      
      fly[12] = 0.0;
      fly[13] = velocity;
      fly[14] = 0.0;
      fly[15] = 1.0;

      bawgl->multInverseVirtualViewMatrix(fly);
    }
  
  // go home
  
  if( BAWGL_PINCH_LEFT_INDEX_RIGHT_INDEX(pinch) && 
      BAWGL_PINCH_LEFT_INDEX_RIGHT_INDEX(pinchChange))
    {
      bawgl->loadVirtualViewMatrix(bawgl->virtualViewHome);
    }

  // set home

  if( BAWGL_PINCH_LEFT_MIDDLE_RIGHT_INDEX(pinch) && 
      BAWGL_PINCH_LEFT_MIDDLE_RIGHT_INDEX(pinchChange))
    {
      bawgl->loadVirtualViewHome(bawgl->virtualViewMatrix);
    }
	  
  // autoview
  
  if( BAWGL_PINCH_LEFT_MIDDLE_RIGHT_INDEX(pinch) && 
      BAWGL_PINCH_LEFT_MIDDLE_RIGHT_INDEX(pinchChange))
    {
      
    }
}

void SCIBaWGLTimer::pick( void )
{
  // Picking
  
    int screenPos[3]; 

  if( BAWGL_PINCH_LEFT_THUMB_LEFT_INDEX(pinch) )
    {
      bawgl->getAllEyePositions();

      if( BAWGL_PINCH_LEFT_THUMB_LEFT_INDEX(pinchChange) )
	{
	  // do the pick here

	  bawgl->getScreenCoordinates(&surfacePinchMatrix[12], screenPos, BAWGL_SURFACE_SPACE);

	  cerr << "screen coords are "<< screenPos[0] <<" and " << screenPos[1]<< "\n";
				 
	  viewwindow->bawgl_pick(BAWGL_PICK_START, screenPos, &surfacePinchMatrix[12]);
	  
	  bawgl->pick = 1;
	} 
      else 
	{
	  // handle pick here
		
	  GLfloat v[3];

	  v[0] = virtualPinchChangeMatrix[12];
	  v[1] = virtualPinchChangeMatrix[13];
	  v[2] = virtualPinchChangeMatrix[14];

	  // bawgl->transformVector(v, BAWGL_SURFACE_SPACE, vv, BAWGL_VIRTUAL_SPACE);

	  viewwindow->bawgl_pick(BAWGL_PICK_MOVE, screenPos, v);
	}
    }

  if( !BAWGL_PINCH_LEFT_THUMB_LEFT_INDEX(pinch) && 
      BAWGL_PINCH_LEFT_THUMB_LEFT_INDEX(pinchChange) )
    {
      GLfloat fv[3];
      
      // unset pinch here
      
      bawgl->pick = 0;
      
      viewwindow->bawgl_pick(BAWGL_PICK_END, screenPos, fv);
    }
}

void SCIBaWGLTimer::run( void )
{
  while( !exit )
    {
      while( !running ) 
        {         
          if( exit ) break; 
#ifdef __sgi
          sginap(0);
#endif
        }
#ifdef __sgi
      usleep(100000);
#endif

      while( running )
        {
          if( exit ) break;

	  // force redraw by send message to salmon

	  if( bawgl->redraw_enable == true )
	    {
	      if( viewwindow->manager->mailbox.numItems() == 0 )
		{
		  viewwindow->manager->mailbox.send(scinew ViewerMessage(viewwindow->id));
		}
	    }

	  // pinch state and change

	  pinch0 = pinch;
	  bawgl->getControllerState(pinchID, &pinch);
	  bawgl->getControllerStateChange(pinchID, &pinch, &pinch0, &pinchChange);
	  
	  // stylus state and change

	  stylus0 = stylus;
	  bawgl->getControllerState(stylusID, &stylus);
	  bawgl->getControllerStateChange(stylusID, &stylus, &stylus0, &stylusChange);

	  // stylus matrices

	  memcpy(realStylus0Matrix, realStylusMatrix, 16*sizeof(GLfloat));

	  bawgl->getControllerMatrix(stylusID, BAWGL_LEFT, 
				     realStylusMatrix, BAWGL_REAL_SPACE);
	  
	  bawgl->transformMatrix(realStylusMatrix, BAWGL_REAL_SPACE,
				 virtualStylusMatrix, BAWGL_VIRTUAL_SPACE);

	  bawgl->getControllerMatrixChange(stylusID, BAWGL_LEFT, realStylusMatrix,
					   realStylus0Matrix, virtualStylusChangeMatrix,
					   BAWGL_REAL_SPACE, BAWGL_VIRTUAL_SPACE);

	  // pinch matrices
	  
	  memcpy(realPinch0Matrix, realPinchMatrix, 16*sizeof(GLfloat));
  
	  bawgl->getControllerMatrix(pinchID, BAWGL_LEFT, 
				     realPinchMatrix, BAWGL_REAL_SPACE);
	  bawgl->transformMatrix(realPinchMatrix, BAWGL_REAL_SPACE,
				 surfacePinchMatrix, BAWGL_SURFACE_SPACE);
	  bawgl->transformMatrix(realPinchMatrix, BAWGL_REAL_SPACE,
				 virtualPinchMatrix, BAWGL_VIRTUAL_SPACE);
	  
	  bawgl->getControllerMatrixChange(stylusID, BAWGL_LEFT, realPinchMatrix,
					   realPinch0Matrix, virtualPinchChangeMatrix,
					   BAWGL_REAL_SPACE, BAWGL_VIRTUAL_SPACE);
	  	  
	  navigate();
	  
	  pick();
	  
#ifdef __sgi
	  usleep(50000);
#endif
        }
    }

  cerr << "Bench event manager quit" << "\n";

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

SCIBaWGL::SCIBaWGL( void )
  : BaWGL()
{ 
  shutting_down = false;
  pick = 0;
}

int SCIBaWGL::start( ViewWindow* r, char* config )
{
  shutting_down = false;

  if( init(config) < 0 ) return(-1);

  timer = new SCIBaWGLTimer(r, this);
  timerthread = new Thread(timer, "BaWGLTimer");
  timerthread->detach();

  timer->start();

  redraw_enable = true;

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
      redraw_enable = false;
    }
}

} // End namespace SCIRun
