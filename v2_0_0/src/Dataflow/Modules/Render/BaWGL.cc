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
 * BaWGL.cc - Semi-Immersive Environment API.
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

#include <math.h>
#include <float.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <Dataflow/Modules/Render/BaWGL.h>

#include <iostream>
using namespace std;

namespace SCIRun {

//------------------------------------ PROTECTED ------------------------------------

void BaWGL::transformVector( GLfloat vfrom[4], int spacefrom, GLfloat vto[4], int spaceto )
{
  GLfloat tmp[4], tmp2[4];

  switch( spacefrom )
    {
    case BAWGL_TRACKER_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  memcpy(vto, vfrom, 4*sizeof(GLfloat));
	  break;
	case BAWGL_REAL_SPACE:
	  glVectorMult(vto, transmitterMatrix, vfrom);
	  break;
	case BAWGL_SURFACE_SPACE:
	  glVectorMult(tmp, transmitterMatrix, vfrom);
	  glVectorMult(vto, surfaceMatrix, tmp);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glVectorMult(tmp, transmitterMatrix, vfrom);
	  glVectorMult(tmp2, surfaceMatrix, tmp);
	  glVectorMult(vto, modelViewMatrix, tmp2);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glVectorMult(tmp, transmitterMatrix, vfrom);
	  tmp[0] /= virtualViewScale;
	  tmp[1] /= virtualViewScale;
	  tmp[2] /= virtualViewScale;
	  glVectorMult(vto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_VIRTUAL_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glVectorMult(tmp, virtualViewMatrix, vfrom);
	  tmp[0] *= virtualViewScale;
	  tmp[1] *= virtualViewScale;
	  tmp[2] *= virtualViewScale;
	  glVectorMult(vto, invTransmitterMatrix, tmp);
	  break;
	case BAWGL_REAL_SPACE:
	  glVectorMult(vto, virtualViewMatrix, vfrom);
	  vto[0] *= virtualViewScale;
	  vto[1] *= virtualViewScale;
	  vto[2] *= virtualViewScale;
	  break;
	case BAWGL_SURFACE_SPACE:
	  glVectorMult(tmp, virtualViewMatrix, vfrom);
	  tmp[0] *= virtualViewScale;
	  tmp[1] *= virtualViewScale;
	  tmp[2] *= virtualViewScale;
	  glVectorMult(vto, surfaceMatrix, tmp);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glVectorMult(tmp, virtualViewMatrix, vfrom);
	  tmp[0] *= virtualViewScale;
	  tmp[1] *= virtualViewScale;
	  tmp[2] *= virtualViewScale;
	  glVectorMult(tmp2, surfaceMatrix, tmp);
	  glVectorMult(vto, modelViewMatrix, tmp2);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  memcpy(vto, vfrom, 4*sizeof(GLfloat));
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_REAL_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glVectorMult(vto, invTransmitterMatrix, vfrom);
	  break;
	case BAWGL_REAL_SPACE:
	  memcpy(vto, vfrom, 4*sizeof(GLfloat));
	  break;
	case BAWGL_SURFACE_SPACE:
	  glVectorMult(vto, surfaceMatrix, vfrom);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glVectorMult(tmp, surfaceMatrix, vfrom);
	  glVectorMult(vto, modelViewMatrix, tmp);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  memcpy(tmp, vfrom, 4*sizeof(GLfloat));
	  tmp[0] /= virtualViewScale;
	  tmp[1] /= virtualViewScale;
	  tmp[2] /= virtualViewScale;
	  glVectorMult(vto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_SURFACE_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glVectorMult(tmp, invSurfaceMatrix, vfrom);
	  glVectorMult(vto, invTransmitterMatrix, tmp);
	  break;
	case BAWGL_REAL_SPACE:
	  glVectorMult(vto, invSurfaceMatrix, vfrom);	  
	  break;
	case BAWGL_SURFACE_SPACE:
	  memcpy(vto, vfrom, 4*sizeof(GLfloat));
	  break;
	case BAWGL_CAMERA_SPACE:
	  glVectorMult(vto, modelViewMatrix, vfrom);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glVectorMult(tmp, invSurfaceMatrix, vfrom);
	  tmp[0] /= virtualViewScale;
	  tmp[1] /= virtualViewScale;
	  tmp[2] /= virtualViewScale;
	  glVectorMult(vto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_CAMERA_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glVectorMult(tmp, invModelViewMatrix, vfrom);
	  glVectorMult(tmp2, invSurfaceMatrix, tmp);
	  glVectorMult(vto, invTransmitterMatrix, tmp2);
	  break;
	case BAWGL_REAL_SPACE:
	  glVectorMult(tmp, invModelViewMatrix, vfrom);
	  glVectorMult(vto, invSurfaceMatrix, tmp);
	  break;
	case BAWGL_SURFACE_SPACE:
	  glVectorMult(vto, invModelViewMatrix, vfrom);
	  break;
	case BAWGL_CAMERA_SPACE:
	  memcpy(vto, vfrom, 4*sizeof(GLfloat));
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glVectorMult(tmp, invModelViewMatrix, vfrom);
	  glVectorMult(tmp2, invSurfaceMatrix, tmp);
	  tmp2[0] /= virtualViewScale;
	  tmp2[1] /= virtualViewScale;
	  tmp2[2] /= virtualViewScale;
	  glVectorMult(vto, invVirtualViewMatrix, tmp2);
	  break;
	default:
	  break;
	}
      break;
    default:
      break;
    }
}

/* Transform matrices between spaces. */

void BaWGL::transformMatrix( GLfloat mfrom[16], int spacefrom, GLfloat mto[16], int spaceto )
{
  GLfloat tmp[16], tmp2[16];

  switch( spacefrom )
    {
    case BAWGL_TRACKER_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  memcpy(mto, mfrom, 16*sizeof(GLfloat));
	  break;
	case BAWGL_REAL_SPACE:
	  glMatrixMult(mto, transmitterMatrix, mfrom);
	  break;
	case BAWGL_SURFACE_SPACE:
	  glMatrixMult(tmp, transmitterMatrix, mfrom);
	  glMatrixMult(mto, surfaceMatrix, tmp);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glMatrixMult(tmp, transmitterMatrix, mfrom);
	  glMatrixMult(tmp2, surfaceMatrix, tmp);
	  glMatrixMult(mto, modelViewMatrix, tmp2);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glMatrixMult(tmp, transmitterMatrix, mfrom);
	  tmp[12] /= virtualViewScale;
	  tmp[13] /= virtualViewScale;
	  tmp[14] /= virtualViewScale;
	  glMatrixMult(mto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_VIRTUAL_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glMatrixMult(tmp, virtualViewMatrix, mfrom);
	  tmp[12] *= virtualViewScale;
	  tmp[13] *= virtualViewScale;
	  tmp[14] *= virtualViewScale;
	  glMatrixMult(mto, invTransmitterMatrix, tmp);
	  break;
	case BAWGL_REAL_SPACE:
	  glMatrixMult(mto, virtualViewMatrix, mfrom);
	  mto[12] *= virtualViewScale;
	  mto[13] *= virtualViewScale;
	  mto[14] *= virtualViewScale;
	  break;
	case BAWGL_SURFACE_SPACE:
	  glMatrixMult(tmp, virtualViewMatrix, mfrom);
	  tmp[12] *= virtualViewScale;
	  tmp[13] *= virtualViewScale;
	  tmp[14] *= virtualViewScale;
	  glMatrixMult(mto, surfaceMatrix, tmp);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glMatrixMult(tmp, virtualViewMatrix, mfrom);
	  tmp[12] *= virtualViewScale;
	  tmp[13] *= virtualViewScale;
	  tmp[14] *= virtualViewScale;
	  glMatrixMult(tmp2, surfaceMatrix, tmp);
	  glMatrixMult(mto, modelViewMatrix, tmp2);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  memcpy(mto, mfrom, 16*sizeof(GLfloat));
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_REAL_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glMatrixMult(mto, invTransmitterMatrix, mfrom);
	  break;
	case BAWGL_REAL_SPACE:
	  memcpy(mto, mfrom, 16*sizeof(GLfloat));
	  break;
	case BAWGL_SURFACE_SPACE:
	  glMatrixMult(mto, surfaceMatrix, mfrom);
	  break;
	case BAWGL_CAMERA_SPACE:
	  glMatrixMult(tmp, surfaceMatrix, mfrom);
	  glMatrixMult(mto, modelViewMatrix, tmp);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  memcpy(tmp, mfrom, 16*sizeof(GLfloat));
	  tmp[12] /= virtualViewScale;
	  tmp[13] /= virtualViewScale;
	  tmp[14] /= virtualViewScale;
	  glMatrixMult(mto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_SURFACE_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glMatrixMult(tmp, invSurfaceMatrix, mfrom);
	  glMatrixMult(mto, invTransmitterMatrix, tmp);
	  break;
	case BAWGL_REAL_SPACE:
	  glMatrixMult(mto, invSurfaceMatrix, mfrom);	  
	  break;
	case BAWGL_SURFACE_SPACE:
	  memcpy(mto, mfrom, 16*sizeof(GLfloat));
	  break;
	case BAWGL_CAMERA_SPACE:
	  glMatrixMult(mto, modelViewMatrix, mfrom);
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glMatrixMult(tmp, invSurfaceMatrix, mfrom);
	  tmp[12] /= virtualViewScale;
	  tmp[13] /= virtualViewScale;
	  tmp[14] /= virtualViewScale;
	  glMatrixMult(mto, invVirtualViewMatrix, tmp);
	  break;
	default:
	  break;
	}
      break;
    case BAWGL_CAMERA_SPACE:
      switch( spaceto )
	{
	case BAWGL_TRACKER_SPACE:
	  glMatrixMult(tmp, invModelViewMatrix, mfrom);
	  glMatrixMult(tmp2, invSurfaceMatrix, tmp);
	  glMatrixMult(mto, invTransmitterMatrix, tmp2);
	  break;
	case BAWGL_REAL_SPACE:
	  glMatrixMult(tmp, invModelViewMatrix, mfrom);
	  glMatrixMult(mto, invSurfaceMatrix, tmp);
	  break;
	case BAWGL_SURFACE_SPACE:
	  glMatrixMult(mto, invModelViewMatrix, mfrom);
	  break;
	case BAWGL_CAMERA_SPACE:
	  memcpy(mto, mfrom, 16*sizeof(GLfloat));
	  break;
	case BAWGL_VIRTUAL_SPACE:
	  glMatrixMult(tmp, invModelViewMatrix, mfrom);
	  glMatrixMult(tmp2, invSurfaceMatrix, tmp);
	  tmp2[12] /= virtualViewScale;
	  tmp2[13] /= virtualViewScale;
	  tmp2[14] /= virtualViewScale;
	  glMatrixMult(mto, invVirtualViewMatrix, tmp2);
	  break;
	default:
	  break;
	}
      break;
    default:
      break;
    }
}

/* Calculate the surface matrix */

void BaWGL::surfaceTransform( void )
{
  GLfloat p[3], tmp[3];

  surfaceTopRight[0] = surfaceWidth/2.0;
  surfaceTopRight[1] = surfaceHeight/2.0;
  surfaceTopRight[2] = 0;

  surfaceBottomLeft[0] = -surfaceTopRight[0];
  surfaceBottomLeft[1] = -surfaceTopRight[1];
  surfaceBottomLeft[2] = 0;

  if( haveSurfaceMatrix == false )
    {
      glAxis2Rot(invSurfaceMatrix, rotAxis, surfaceAngle/180.0*M_PI);
      invSurfaceMatrix[12] = rotCenter[0];
      invSurfaceMatrix[13] = rotCenter[1];
      invSurfaceMatrix[14] = rotCenter[2];

      p[0] = -rotCenter[0];
      p[1] = -rotCenter[1];
      p[2] = -rotCenter[2];
  
      glTransform(tmp, invSurfaceMatrix, p);

      invSurfaceMatrix[12] = tmp[0];
      invSurfaceMatrix[13] = tmp[1];
      invSurfaceMatrix[14] = tmp[2];
      
      invSurfaceMatrix[3] = 0.0;
      invSurfaceMatrix[7] = 0.0;
      invSurfaceMatrix[11] = 0.0;
      invSurfaceMatrix[15] = 1.0;

      glInverse(surfaceMatrix, invSurfaceMatrix);
    }
  else
    {
      glInverse(invSurfaceMatrix, surfaceMatrix);
    }
}

/* Calculate frustum left bottom and top right points on the near clipping plane. */

void BaWGL::frustum( GLfloat m[16], GLfloat r0[3], GLfloat r[3], GLfloat n )
{
  GLfloat z = m[2]*r0[0] + m[6]*r0[1] + m[10]*r0[2] + m[14];
  
  r[0] = (m[0]*r0[0] + m[4]*r0[1] + m[8]*r0[2] + m[12])* -n/z;
  r[1] = (m[1]*r0[0] + m[5]*r0[1] + m[9]*r0[2] + m[13])* -n/z;
  r[2] = -n;
}

void BaWGL::scaleEyeOffsets( void )
{
  scaledEyeOffset[BAWGL_MIDDLE_EYE][0] = eyeOffset[BAWGL_MIDDLE_EYE][0];
  scaledEyeOffset[BAWGL_MIDDLE_EYE][1] = eyeOffset[BAWGL_MIDDLE_EYE][1];
  scaledEyeOffset[BAWGL_MIDDLE_EYE][2] = eyeOffset[BAWGL_MIDDLE_EYE][2];
  
  scaledEyeOffset[BAWGL_LEFT_EYE][0] = (eyeOffset[BAWGL_LEFT_EYE][0] - eyeOffset[BAWGL_MIDDLE_EYE][0]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][0];
  scaledEyeOffset[BAWGL_LEFT_EYE][1] = (eyeOffset[BAWGL_LEFT_EYE][1] - eyeOffset[BAWGL_MIDDLE_EYE][1]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][1];
  scaledEyeOffset[BAWGL_LEFT_EYE][2] = (eyeOffset[BAWGL_LEFT_EYE][2] - eyeOffset[BAWGL_MIDDLE_EYE][2]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][2];

  scaledEyeOffset[BAWGL_RIGHT_EYE][0] = (eyeOffset[BAWGL_RIGHT_EYE][0] - eyeOffset[BAWGL_MIDDLE_EYE][0]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][0];
  scaledEyeOffset[BAWGL_RIGHT_EYE][1] = (eyeOffset[BAWGL_RIGHT_EYE][1] - eyeOffset[BAWGL_MIDDLE_EYE][1]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][1];
  scaledEyeOffset[BAWGL_RIGHT_EYE][2] = (eyeOffset[BAWGL_RIGHT_EYE][2] - eyeOffset[BAWGL_MIDDLE_EYE][2]) 
    / virtualViewScale + eyeOffset[BAWGL_MIDDLE_EYE][2];
}

//------------------------------------ PUBLIC ------------------------------------

BaWGL::BaWGL( void )
{
}

BaWGL::~BaWGL( void )
{
}

/* Initialize shared memory structure based on arena files given. */

int BaWGL::init( char* config )
{
  if( parse(config) < 0 )
    {
      cerr << "Error: Parsing configuration file " << config << "." << endl;
      return(-1);
    }
  else
    {
      glInverse(invTransmitterMatrix, transmitterMatrix);
      surfaceTransform();
      glEye(virtualViewMatrix);
      glEye(invVirtualViewMatrix);
      virtualViewScale = 1.0;
      virtualViewScaleMin = 0.0;
      virtualViewScaleMax = FLT_MAX;
      glEye(virtualViewHome);
      // scaleEyeOffsets();

      if( tracker.shmem.attach(tracker.arena, &(tracker.data)) < 0) return(-1);

      for( int i=0; i<NUM_CONTROLLERS; i++ )
	{
	  switch( controller[i].type )
	    {
	    case CONTROLLER_PINCH:
	      if( controller[i].shmem.attach(controller[i].arena, &(controller[i].data)) < 0 )
		return(-1);
	      break;
	    case CONTROLLER_STYLUS:
	      break;
	    case CONTROLLER_I3STICK:
	      break;
	    case CONTROLLER_NONE:
	      break;
	    }
	}

      system(windowInitCmd);
      
      return(0);
    }
}

/* Detach shared memory structure. */

void BaWGL::quit( void )
{
  system(windowExitCmd);
    
  tracker.shmem.detach();

  for( int i=0; i<NUM_CONTROLLERS; i++ )
    {
      switch( controller[i].type )
	{
	case CONTROLLER_PINCH:
	  controller[i].shmem.detach();
	  break;
	case CONTROLLER_STYLUS:
	  break;
	case CONTROLLER_I3STICK:
	  break;
	case CONTROLLER_NONE:
	  break;
	}
    }
}

/* Get controller id for a given type */

int BaWGL::getControllerID( int type )
{
  int i, id;

  for( i=0; i<NUM_CONTROLLERS; i++ )
    if( controller[i].type == type ) id = i;

  return(id);
}

/* space: TRACKER, REAL, SURFACE, MODEL, CAMERA, but not SCREEN */
//requires that the eyepos, modelview and projections have been set

void BaWGL::getTrackerMatrix( int id, GLfloat m[16], int space )
{
  GLfloat tmp[16];

  if( tracker.data )
      {

	switch( tracker.type )
	  {
	  case TRACKER_FASTRAK:
	    if ( id >= 0 && id < FASTRAK_MAX_RECEIVER )
	      memcpy(tmp, ((Fastrak*)(tracker.data))->receiver[id].m, 16*sizeof(GLfloat));
	    break;
	  case TRACKER_FOB:
	    if( id >= 0 && id < FOB_MAX_RECEIVER )
	      memcpy(tmp, ((FoB*)(tracker.data))->receiver[id].m, 16*sizeof(GLfloat));
	    break;
	  case TRACKER_NONE:
	    break;
	  default:
	    break;
	  }

	transformMatrix(tmp, BAWGL_TRACKER_SPACE, m, space);
      }
}


void BaWGL::getControllerState( int id, void* s )
{
  if( tracker.data )
      {
	switch( controller[id].type )
	  {
	  case CONTROLLER_STYLUS:
	    switch( tracker.type )
	      {
	      case TRACKER_FASTRAK:
		*(Stylus*)s = ((Fastrak*)(tracker.data))->stylus;
		break;
	      case TRACKER_FOB:
	      case TRACKER_NONE:
	      default:
		break;
	      }
	    break;
	  case CONTROLLER_PINCH:
	    *(Gesture*)s = ((Pinch*)(controller[id].data))->gesture;
	    break;
	  case CONTROLLER_I3STICK:
	    break;
	  case CONTROLLER_NONE:
	    break;
	  default:
	    break;
	  }
      }
}


void BaWGL::getControllerStateChange( int id, void* s, void* ps, void* c )
{
  switch( controller[id].type )
    {
    case CONTROLLER_STYLUS:
      switch( tracker.type )
	{
	case TRACKER_FASTRAK:
	  *(Stylus*)c = *(Stylus*)ps ^ *(Stylus*)s;
	  break;
	case TRACKER_FOB:
	case TRACKER_NONE:
	default:
	  break;
	}
      break;
    case CONTROLLER_PINCH:
      *(Gesture*)c = *(Gesture*)ps ^ *(Gesture*)s;
      break;
    case CONTROLLER_I3STICK:
      break;
    case CONTROLLER_NONE:
      break;
    default:
      break;
    }
}


void BaWGL::getRelTransform( GLfloat m[16], GLfloat mfrom[16], GLfloat mto[16], 
			     int spacefrom, int spaceto )
{
  GLfloat R[16], k[3], k0[3], theta, tmp[16];

  R[0] = mfrom[0]*mto[0] + mfrom[1]*mto[1] + mfrom[2]*mto[2];
  R[1] = mfrom[4]*mto[0] + mfrom[5]*mto[1] + mfrom[6]*mto[2];
  R[2] = mfrom[8]*mto[0] + mfrom[9]*mto[1] + mfrom[10]*mto[2];
  R[3] = 0.0;

  R[4] = mfrom[0]*mto[4] + mfrom[1]*mto[5] + mfrom[2]*mto[6];
  R[5] = mfrom[4]*mto[4] + mfrom[5]*mto[5] + mfrom[6]*mto[6];
  R[6] = mfrom[8]*mto[4] + mfrom[9]*mto[5] + mfrom[10]*mto[6];
  R[7] = 0.0;

  R[8] = mfrom[0]*mto[8] + mfrom[1]*mto[9] + mfrom[2]*mto[10];
  R[9] = mfrom[4]*mto[8] + mfrom[5]*mto[9] + mfrom[6]*mto[10];
  R[10] = mfrom[8]*mto[8] + mfrom[9]*mto[9] + mfrom[10]*mto[10];
  R[11] = 0.0;

  R[12] = 0.0;
  R[13] = 0.0;
  R[14] = 0.0;
  R[15] = 1.0;
  
  glRot2Axis(k, &theta, R);

  k0[0] = mfrom[0]*k[0] + mfrom[4]*k[1] + mfrom[8]*k[2];
  k0[1] = mfrom[1]*k[0] + mfrom[5]*k[1] + mfrom[9]*k[2];
  k0[2] = mfrom[2]*k[0] + mfrom[6]*k[1] + mfrom[10]*k[2];
  
  glEye(tmp);
  transformMatrix(tmp, spacefrom, R, spaceto);
  
  k[0] = R[0]*k0[0] + R[4]*k0[1] + R[8]*k0[2];
  k[1] = R[1]*k0[0] + R[5]*k0[1] + R[9]*k0[2];
  k[2] = R[2]*k0[0] + R[6]*k0[1] + R[10]*k0[2];
  
  glAxis2Rot(m, k, theta);

  k0[0] = mto[12] - mfrom[12];
  k0[1] = mto[13] - mfrom[13];
  k0[2] = mto[14] - mfrom[14];

  m[12] = R[0]*k0[0] + R[4]*k0[1] + R[8]*k0[2];
  m[13] = R[1]*k0[0] + R[5]*k0[1] + R[9]*k0[2];
  m[14] = R[2]*k0[0] + R[6]*k0[1] + R[10]*k0[2];
  m[15] = 1.0;

  if( spaceto == BAWGL_VIRTUAL_SPACE )
    {
      m[12] /= virtualViewScale;
      m[13] /= virtualViewScale;
      m[14] /= virtualViewScale;
    }

  m[3] = 0.0; m[7] = 0.0; m[11] = 0.0;
}


void BaWGL::getControllerMatrix( int id, int rid, GLfloat m[16], int space )
{
  GLfloat tmp[16], tmp2[16];

  if( tracker.data )
      {
	switch( controller[id].type )
	  {
	  case CONTROLLER_STYLUS:
	    switch( tracker.type )
	      {
	      case TRACKER_FASTRAK:
		memcpy(tmp, 
		       ((Fastrak*)(tracker.data))->receiver[controller[id].receiver[0]].m, 
		       16*sizeof(GLfloat));
		glMatrixMult(tmp2, tmp, controller[id].offset[rid]);
		transformMatrix(tmp2, BAWGL_TRACKER_SPACE, m, space);
		break;
	      case TRACKER_FOB:
		memcpy(tmp, 
		       ((FoB*)(tracker.data))->receiver[controller[id].receiver[0]].m, 
		       16*sizeof(GLfloat));
		glMatrixMult(tmp2, tmp, controller[id].offset[rid]);
		transformMatrix(tmp2, BAWGL_TRACKER_SPACE, m, space);
		break;
	      case TRACKER_NONE:
	      default:
		break;
	      }
	    break;
	  case CONTROLLER_PINCH:
	    switch( tracker.type )
	      {
	      case TRACKER_FASTRAK:
		memcpy(tmp, 
		       ((Fastrak*)(tracker.data))->receiver[controller[id].receiver[rid]].m,
		       16*sizeof(GLfloat));
		glMatrixMult(tmp2, tmp, controller[id].offset[rid]);
		transformMatrix(tmp2, BAWGL_TRACKER_SPACE, m, space);
		break;
	      case TRACKER_FOB:
		memcpy(tmp, 
		       ((FoB*)(tracker.data))->receiver[controller[id].receiver[rid]].m,
		       16*sizeof(GLfloat));
		glMatrixMult(tmp2, tmp, controller[id].offset[rid]);
		transformMatrix(tmp2, BAWGL_TRACKER_SPACE, m, space);
		break;
	      case TRACKER_NONE:
	      default:
		break;
	      }
	    break;
	  case CONTROLLER_I3STICK:
	    break;
	  case CONTROLLER_NONE:
	  default:
	    break;
	  }
      }
}


void BaWGL::getControllerMatrixChange( int , int , GLfloat m[16], GLfloat pm[16], 
				       GLfloat mc[16], int spacefrom, int spaceto )
{
  getRelTransform(mc, pm, m, spacefrom, spaceto);
}

/* Calculate eye positions from tracker data. */

void BaWGL::getEyePosition( int eye )
{
  GLfloat m[16], tmp[3], tmp2[3];

  if( tracker.data ) 
    {
      switch( tracker.type )
	{
	case TRACKER_FASTRAK:
	  memcpy(m, ((Fastrak*)(tracker.data))->receiver[eyeReceiver].m, 16*sizeof(GLfloat));
	  break;
	case TRACKER_FOB:
	  memcpy(m, ((FoB*)(tracker.data))->receiver[eyeReceiver].m, 16*sizeof(GLfloat));
	  break;
	case TRACKER_NONE:
	default:
	  break;
	}
  
      glTransform(tmp, m, eyeOffset[eye]);
      glTransform(tmp2, transmitterMatrix, tmp);
      glTransform(eyePosition[eye], surfaceMatrix, tmp2);
    }
}

/* Calculate eye positions from tracker data. */

void BaWGL::getAllEyePositions( void )
{
  GLfloat m[16], tmp[3], tmp2[3];
  
  if( tracker.data )
    {

      switch( tracker.type )
	{
	case TRACKER_FASTRAK:
	  memcpy(m, ((Fastrak*)(tracker.data))->receiver[eyeReceiver].m, 16*sizeof(GLfloat));
	  break;
	case TRACKER_FOB:
	  memcpy(m, ((FoB*)(tracker.data))->receiver[eyeReceiver].m, 16*sizeof(GLfloat));
	  break;
	case TRACKER_NONE:
	default:
	  break;
	}
      
      glTransform(tmp, m, eyeOffset[BAWGL_LEFT_EYE]);
      glTransform(tmp2, transmitterMatrix, tmp);
      glTransform(eyePosition[BAWGL_LEFT_EYE], surfaceMatrix, tmp2);

      glTransform(tmp, m, eyeOffset[BAWGL_MIDDLE_EYE]);
      glTransform(tmp2, transmitterMatrix, tmp);
      glTransform(eyePosition[BAWGL_MIDDLE_EYE], surfaceMatrix, tmp2);

      glTransform(tmp, m, eyeOffset[BAWGL_RIGHT_EYE]);
      glTransform(tmp2, transmitterMatrix, tmp);
      glTransform(eyePosition[BAWGL_RIGHT_EYE], surfaceMatrix, tmp2);
    }
}


/* Set modelview matrix for the specified eye. */

void BaWGL::setModelViewMatrix( int eye )
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eyePosition[eye][0], eyePosition[eye][1], eyePosition[eye][2],
	    eyePosition[eye][0], eyePosition[eye][1], 0.0,
	    upVector[0], upVector[1], upVector[2]);
  glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix);
  glInverse(invModelViewMatrix, modelViewMatrix);
}

/* Set projection matrix for the specified eye. */

void BaWGL::setProjectionMatrix( int eye )
{
  frustum(modelViewMatrix, surfaceBottomLeft, bottomLeft[eye], nearClip[eye]);
  frustum(modelViewMatrix, surfaceTopRight, topRight[eye], nearClip[eye]);
    
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(bottomLeft[eye][0], topRight[eye][0],
	    bottomLeft[eye][1], topRight[eye][1],
	    nearClip[eye], farClip[eye]);
  glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);
  glMatrixMode(GL_MODELVIEW);
}

void BaWGL::setViewPort( GLint xl, GLint yl, GLint xr, GLint yr )
{
  viewPort[0] = xl;
  viewPort[1] = yl;
  viewPort[2] = xr;
  viewPort[3] = yr;
}

void BaWGL::setPickProjectionMatrix( int eye, GLint x, GLint y, GLfloat pickwin )
{
  frustum(modelViewMatrix, surfaceBottomLeft, bottomLeft[eye], nearClip[eye]);
  frustum(modelViewMatrix, surfaceTopRight, topRight[eye], nearClip[eye]);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPickMatrix(x, y, pickwin, pickwin, viewPort);
  glFrustum(bottomLeft[eye][0], topRight[eye][0],
	    bottomLeft[eye][1], topRight[eye][1],
	    nearClip[eye], farClip[eye]);
  
  glMatrixMode(GL_MODELVIEW);
}

void BaWGL::setNearFar( int eye, GLfloat p[3], GLfloat d )
{
  // assume that p[3] is in camera coords, and d is a percentage of the z val
  // assume we mean to set the middle eye!!

  GLfloat near, far;

  near = p[3] + (p[3]*d);
  far = p[3] - (p[3]*d);
    
  frustum(modelViewMatrix, surfaceBottomLeft, bottomLeft[eye], nearClip[eye]);
  frustum(modelViewMatrix, surfaceTopRight, topRight[eye], nearClip[eye]);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(bottomLeft[eye][0], topRight[eye][0],
	    bottomLeft[eye][1], topRight[eye][1],
	    near, far);
  glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);
  glMatrixMode(GL_MODELVIEW);
}

void BaWGL::getScreenCoordinates( GLfloat p[3], int s[3], int space )
{
  GLdouble win[3], modelView[16], projection[16];
  GLfloat pi[4], po[4];

  pi[0] = p[0];
  pi[1] = p[1];
  pi[2] = p[2];
  pi[3] = 1.0;

  transformVector(pi, space, po, BAWGL_SURFACE_SPACE); 

  for( int i=0; i<16; i++ )
    {
      modelView[i] = (GLdouble)modelViewMatrix[i];
      projection[i] = (GLdouble)projectionMatrix[i];
    }

  gluProject((GLdouble)po[0], (GLdouble)po[1], (GLdouble)po[2],
	     modelView, projection, viewPort,
	     &win[0], &win[1], &win[2]);

  s[0] = (int)win[0];
  s[1] = (int)win[1];
  s[2] = (int)win[2];
}

void BaWGL::setSurfaceView()
{
  glMultMatrixf(surfaceMatrix);
}

/* Multiply the OpenGL modelview matrix by the virtualView matrix. */  

void BaWGL::setVirtualView()
{
  glScalef(virtualViewScale, virtualViewScale, virtualViewScale);
  glMultMatrixf(virtualViewMatrix);
}

/* Functions to change the virtualview - inverse virtualview matrices */

void BaWGL::getVirtualViewMatrix( GLfloat m[16] )
{
  memcpy(m, virtualViewMatrix, 16*sizeof(GLfloat));
}

void BaWGL::getInverseVirtualViewMatrix( GLfloat m[16] )
{
  memcpy(m, invVirtualViewMatrix, 16*sizeof(GLfloat));
}

void BaWGL::loadVirtualViewMatrix( GLfloat m[16] )
{
  memcpy(virtualViewMatrix, m, 16*sizeof(GLfloat));
  glInverse(invVirtualViewMatrix, virtualViewMatrix);
}

void BaWGL::loadInverseVirtualViewMatrix( GLfloat m[16] )
{
  memcpy(invVirtualViewMatrix, m, 16*sizeof(GLfloat));
  glInverse(virtualViewMatrix, invVirtualViewMatrix);
}

void BaWGL::loadVirtualViewHome(GLfloat m[16])
{
  memcpy(virtualViewHome, m, 16*sizeof(GLfloat));
}

void BaWGL::multVirtualViewMatrix( GLfloat m[16] )
{
  GLfloat tmp[16];

  glMatrixMult(tmp, virtualViewMatrix, m);
  memcpy(virtualViewMatrix, tmp, 16*sizeof(GLfloat));
  glInverse(invVirtualViewMatrix, virtualViewMatrix);
}

void BaWGL::multInverseVirtualViewMatrix( GLfloat m[16] )
{
  GLfloat tmp[16];

  glMatrixMult(tmp, invVirtualViewMatrix, m);
  memcpy(invVirtualViewMatrix, tmp, 16*sizeof(GLfloat));
  glInverse(virtualViewMatrix, invVirtualViewMatrix);
}

GLfloat BaWGL::getVirtualViewScale( void )
{
  return(virtualViewScale);
}

void BaWGL::loadVirtualViewScale( GLfloat s )
{
  if( s < virtualViewScaleMin )
    { 
      virtualViewScale = virtualViewScaleMin;
    }
  else if( s > virtualViewScaleMax )
    {
      virtualViewScale = virtualViewScaleMax;
    }
  else virtualViewScale = s;

  // scaleEyeOffsets();
}

void BaWGL::multVirtualViewScale( GLfloat s )
{
  virtualViewScale *= s;

  if( virtualViewScale < virtualViewScaleMin )
    { 
      virtualViewScale = virtualViewScaleMin;
    }
  else if( virtualViewScale > virtualViewScaleMax )
    {
      virtualViewScale = virtualViewScaleMax;
    }

  // scaleEyeOffsets();
}

void BaWGL::addVirtualViewScale( GLfloat s )
{
  virtualViewScale += s;

  if( virtualViewScale < virtualViewScaleMin )
    { 
      virtualViewScale = virtualViewScaleMin;
    }
  else if( virtualViewScale > virtualViewScaleMax )
    {
      virtualViewScale = virtualViewScaleMax;
    }

  // scaleEyeOffsets(); 
}

void BaWGL::setVirtualViewScaleLimits( GLfloat min, GLfloat max )
{
  virtualViewScaleMin = min;
  virtualViewScaleMax = max;
}

} // End namespace SCIRun
