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

/*------------------------------------------------------------------
 * BaWGL.h - Semi-Immersive Environment API.
 *
 * Author: Milan Ikits (ikits@cs.utah.edu)
 *         Joe Kniss   (jmk@cs.utah.edu)
 *
 * Copyright (C) 1999 
 * Center for Scientific Computing and Imaging
 * University of Utah
 * 
 * Creation: 8/21/99
 * Last modification:
 * Comments:
 *------------------------------------------------------------------*/

#ifndef BAWGL_H_
#define BAWGL_H_

#include <GL/gl.h>
#include <fstream>

#include <Dataflow/Modules/Render/SharedMemory.h>
#include <Dataflow/Modules/Render/Tracker.h>
#include <Dataflow/Modules/Render/Controller.h>

#include <Dataflow/Modules/Render/fastrak.h>
#include <Dataflow/Modules/Render/fob.h>
#include <Dataflow/Modules/Render/pinch.h>
#include <Dataflow/Modules/Render/glMath.h>

#define BAWGL_STYLUS CONTROLLER_STYLUS
#define BAWGL_PINCH CONTROLLER_PINCH

#define BAWGL_ONE CONTROLLER_ONE
#define BAWGL_LEFT CONTROLLER_LEFT
#define BAWGL_RIGHT CONTROLLER_RIGHT

#define BAWGL_TRACKER_SPACE 0
#define BAWGL_REAL_SPACE    1
#define BAWGL_SURFACE_SPACE 2
#define BAWGL_CAMERA_SPACE  3
#define BAWGL_SCREEN_SPACE  4
#define BAWGL_MODEL_SPACE   5
#define BAWGL_VIRTUAL_SPACE BAWGL_MODEL_SPACE

#define BAWGL_PICK_START 0
#define BAWGL_PICK_MOVE 1
#define BAWGL_PICK_END 2

#define BAWGL_LEFT_EYE 0
#define BAWGL_MIDDLE_EYE 1
#define BAWGL_RIGHT_EYE 2

#define BAWGL_STYLUS_ON STYLUS_ON
#define BAWGL_STYLUS_OFF STYLUS_OFF

#define BAWGL_PINCH_NONE PINCH_NONE

#define BAWGL_PINCH_LEFT_THUMB_LEFT_INDEX(x)  (((x) & PINCH_LEFT_MASK) == PINCH_LEFT_THUMB_LEFT_INDEX)
#define BAWGL_PINCH_LEFT_THUMB_LEFT_MIDDLE(x) (((x) & PINCH_LEFT_MASK) == PINCH_LEFT_THUMB_LEFT_MIDDLE)
#define BAWGL_PINCH_LEFT_THUMB_LEFT_RING(x)   (((x) & PINCH_LEFT_MASK) == PINCH_LEFT_THUMB_LEFT_RING)
#define BAWGL_PINCH_LEFT_THUMB_LEFT_PINKY(x)  (((x) & PINCH_LEFT_MASK) == PINCH_LEFT_THUMB_LEFT_PINKY)

#define BAWGL_PINCH_RIGHT_THUMB_RIGHT_INDEX(x)  (((x) & PINCH_RIGHT_MASK) == PINCH_RIGHT_THUMB_RIGHT_INDEX)
#define BAWGL_PINCH_RIGHT_THUMB_RIGHT_MIDDLE(x) (((x) & PINCH_RIGHT_MASK) == PINCH_RIGHT_THUMB_RIGHT_MIDDLE)
#define BAWGL_PINCH_RIGHT_THUMB_RIGHT_RING(x)   (((x) & PINCH_RIGHT_MASK) == PINCH_RIGHT_THUMB_RIGHT_RING)
#define BAWGL_PINCH_RIGHT_THUMB_RIGHT_PINKY(x)  (((x) & PINCH_RIGHT_MASK) == PINCH_RIGHT_THUMB_RIGHT_PINKY)

#define BAWGL_PINCH_LEFT_THUMB_RIGHT_THUMB(x)  ((x) & PINCH_LEFT_THUMB_RIGHT_THUMB)
#define BAWGL_PINCH_LEFT_THUMB_RIGHT_INDEX(x)  ((x) & PINCH_LEFT_THUMB_RIGHT_INDEX)
#define BAWGL_PINCH_LEFT_THUMB_RIGHT_MIDDLE(x) ((x) & PINCH_LEFT_THUMB_RIGHT_MIDDLE)
#define BAWGL_PINCH_LEFT_THUMB_RIGHT_RING(x)   ((x) & PINCH_LEFT_THUMB_RIGHT_RING)
#define BAWGL_PINCH_LEFT_THUMB_RIGHT_PINKY(x)  ((x) & PINCH_LEFT_THUMB_RIGHT_PINKY)

#define BAWGL_PINCH_LEFT_INDEX_RIGHT_THUMB(x)  ((x) & PINCH_LEFT_INDEX_RIGHT_THUMB)
#define BAWGL_PINCH_LEFT_INDEX_RIGHT_INDEX(x)  ((x) & PINCH_LEFT_INDEX_RIGHT_INDEX)
#define BAWGL_PINCH_LEFT_INDEX_RIGHT_MIDDLE(x) ((x) & PINCH_LEFT_INDEX_RIGHT_MIDDLE)
#define BAWGL_PINCH_LEFT_INDEX_RIGHT_RING(x)   ((x) & PINCH_LEFT_INDEX_RIGHT_RING)
#define BAWGL_PINCH_LEFT_INDEX_RIGHT_PINKY(x)  ((x) & PINCH_LEFT_INDEX_RIGHT_PINKY)

#define BAWGL_PINCH_LEFT_MIDDLE_RIGHT_THUMB(x)  ((x) & PINCH_LEFT_MIDDLE_RIGHT_THUMB)
#define BAWGL_PINCH_LEFT_MIDDLE_RIGHT_INDEX(x)  ((x) & PINCH_LEFT_MIDDLE_RIGHT_INDEX)
#define BAWGL_PINCH_LEFT_MIDDLE_RIGHT_MIDDLE(x) ((x) & PINCH_LEFT_MIDDLE_RIGHT_MIDDLE)
#define BAWGL_PINCH_LEFT_MIDDLE_RIGHT_RING(x)   ((x) & PINCH_LEFT_MIDDLE_RIGHT_RING)
#define BAWGL_PINCH_LEFT_MIDDLE_RIGHT_PINKY(x)  ((x) & PINCH_LEFT_MIDDLE_RIGHT_PINKY)

#define BAWGL_PINCH_LEFT_RING_RIGHT_THUMB(x)  ((x) & PINCH_LEFT_RING_RIGHT_THUMB)
#define BAWGL_PINCH_LEFT_RING_RIGHT_INDEX(x)  ((x) & PINCH_LEFT_RING_RIGHT_INDEX)
#define BAWGL_PINCH_LEFT_RING_RIGHT_MIDDLE(x) ((x) & PINCH_LEFT_RING_RIGHT_MIDDLE)
#define BAWGL_PINCH_LEFT_RING_RIGHT_RING(x)   ((x) & PINCH_LEFT_RING_RIGHT_RING)
#define BAWGL_PINCH_LEFT_RING_RIGHT_PINKY(x)  ((x) & PINCH_LEFT_RING_RIGHT_PINKY)

#define BAWGL_PINCH_LEFT_PINKY_RIGHT_THUMB(x)  ((x) & PINCH_LEFT_PINKY_RIGHT_THUMB)
#define BAWGL_PINCH_LEFT_PINKY_RIGHT_INDEX(x)  ((x) & PINCH_LEFT_PINKY_RIGHT_INDEX)
#define BAWGL_PINCH_LEFT_PINKY_RIGHT_MIDDLE(x) ((x) & PINCH_LEFT_PINKY_RIGHT_MIDDLE)
#define BAWGL_PINCH_LEFT_PINKY_RIGHT_RING(x)   ((x) & PINCH_LEFT_PINKY_RIGHT_RING)
#define BAWGL_PINCH_LEFT_PINKY_RIGHT_PINKY(x)  ((x) & PINCH_LEFT_PINKY_RIGHT_PINKY)

#define NUM_CONTROLLERS 3
#define CMDLEN 256

namespace SCIRun {
using namespace std;

class BaWGL {

public:
	
/* TRACKER_SPACE <-> REAL_SPACE */
  GLfloat transmitterMatrix[16], invTransmitterMatrix[16];
	
/* REAL_SPACE <-> SURFACE_SPACE */
  GLfloat surfaceMatrix[16], invSurfaceMatrix[16];
  GLfloat rotCenter[3], rotAxis[3];
  GLfloat surfaceAngle;
  bool haveSurfaceMatrix;

/* SURFACE_SPACE <-> CAMERA_SPACE */
  GLfloat eyeOffset[3][3]; /* left, middle, right */
  GLfloat scaledEyeOffset[3][3];

  int eyeReceiver;
  GLfloat eyePosition[3][3];
  GLfloat upVector[3];

  GLfloat virtualViewHome[16];

  GLfloat surfaceWidth, surfaceHeight;
  GLfloat surfaceBottomLeft[3], surfaceTopRight[3];

  GLfloat bottomLeft[3][3]; /* left, middle, right */
  GLfloat topRight[3][3];   /* left, middle, right */

  GLfloat nearClip[3], farClip[3]; /* left, middle, right */

  GLfloat modelViewMatrix[16], invModelViewMatrix[16];
  GLfloat projectionMatrix[16];

  GLint viewPort[4];

/* REAL_SPACE <-> MODEL_SPACE */
  GLfloat virtualViewMatrix[16];
  GLfloat invVirtualViewMatrix[16];
  GLfloat virtualViewScale, virtualViewScaleMin, virtualViewScaleMax;

/* Tracker */
  Tracker tracker;

/* Controllers */
  Controller controller[NUM_CONTROLLERS];

/* Window parameters */
  char windowInitCmd[CMDLEN];
  char windowExitCmd[CMDLEN];

/* Transform vectors or matrices between spaces */
  void transformVector( GLfloat vfrom[4], int spacefrom, GLfloat vto[4], int spaceto );
  void transformMatrix( GLfloat mfrom[16], int spacefrom, GLfloat mto[16], int spaceto );
  
/* Calculate the surface matrix */
  void surfaceTransform( void );

  void getRelTransform( GLfloat m[16], GLfloat mfrom[16], GLfloat mto[16], int spacefrom, int spaceto );

/* Calculate frustum left bottom and top right points on the near clipping plane */
  void frustum( GLfloat m[16], GLfloat r0[3], GLfloat r[3], GLfloat n );

  void scaleEyeOffsets( void );

/* Parse config files */
  int parse( char* fname );
  int parseData( ifstream& f, char* fname, int& l, int t, int cnum = 0, int rnum = 0 );

public:

  BaWGL( void );
  virtual ~BaWGL( void );

/* Initialize shared memory structure based on arena files given */
  int init( char* );

/* Detach shared memory structure */
  void quit( void );

/* Get controller id for a given type */
  int getControllerID( int type );

/* Return tracker matrix for a given receiver */
  void getTrackerMatrix( int id, GLfloat m[16], int space );

/* Return state and whether state change happened for a given controller */
  void getControllerState( int id, void* s );
  void getControllerStateChange( int id, void* s, void* ps, void* sc );

/* Return tracker matrix for a given controller */
  void getControllerMatrix( int id, int rid, GLfloat m[16], int space );
  void getControllerMatrixChange( int id, int rid, GLfloat m[16], GLfloat pm[16], GLfloat mc[16], int spacefrom, int spaceto );

/* Calculate eye positions from tracker data */
  void getEyePosition( int eye );
  void getAllEyePositions( void );

/* Set viewport */
  void setViewPort( GLint xl, GLint yl, GLint xr, GLint yr );

/* Set modelview and projection matrices for one eye */
/* SHOULD BE USED IN THE OPENGL THREAD ONLY!!! */
  void setModelViewMatrix( int eye );
  void setProjectionMatrix( int eye );
  
/* Functions to support picking */
/* SHOULD BE USED IN THE OPENGL THREAD ONLY!!! */
  void setPickProjectionMatrix( int eye, GLint x, GLint y, GLfloat pickwin );
  void setNearFar( int eye, GLfloat p[3], GLfloat d );

/* Functions to support picking */
  void getScreenCoordinates( GLfloat p[3], int s[3], int space );

/* Multiply the modelview matrix by the surface matrix */
/* SHOULD BE USED IN THE OPENGL THREAD ONLY!!! */
  void setSurfaceView( void );

/* Multiply the modelview matrix by virtualview matrix */
/* SHOULD BE USED IN THE OPENGL THREAD ONLY!!! */
  void setVirtualView( void );
  
/* Manipulate the virtual view matrix */
  void getVirtualViewMatrix( GLfloat m[16] );
  void getInverseVirtualViewMatrix( GLfloat m[16] );
  void loadVirtualViewMatrix( GLfloat m[16] );
  void loadInverseVirtualViewMatrix( GLfloat m[16] );
  void multVirtualViewMatrix( GLfloat m[16] );
  void multInverseVirtualViewMatrix( GLfloat m[16] );

  void loadVirtualViewHome(GLfloat m[16]);

/* Manipulate the virtual view scale */  
  GLfloat getVirtualViewScale( void );
  void loadVirtualViewScale( GLfloat s );
  void multVirtualViewScale( GLfloat s );
  void addVirtualViewScale( GLfloat s );

/* Set virtual view scale limits */
  void setVirtualViewScaleLimits( GLfloat min, GLfloat max );
};

} // End namespace SCIRun

#endif /* BAWGL_H_ */
