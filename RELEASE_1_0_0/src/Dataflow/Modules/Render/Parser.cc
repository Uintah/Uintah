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
 * Parser.cc - Simple parser for sie configuration files.
 * 
 * Author: Milan Ikits (ikits@cs.utah.edu)
 * 
 * Copyright (C) 1999
 * Center for Scientific Computing and Imaging
 * University of Utah
 *
 * Creation: 08/05/99
 * Last modification: 08/05/99
 * Comments: Has to be replaced w/ a real parser...
 *------------------------------------------------------------------*/


#include <iostream>
using std::cerr;
using std::endl;

#include <fstream>
using namespace std;

#include <stdio.h>

#include <string.h>
#include <Dataflow/Modules/Render/BaWGL.h>

const int TOKENSIZE = 1024;

const int TRACKER_TYPE                 = 10;
const int TRACKER_SHARED_ARENA         = 11;
const int TRACKER_TRANSMITTER_MATRIX   = 12;

const int VIEWER_LEFT_EYE_OFFSET       = 20;
const int VIEWER_MIDDLE_EYE_OFFSET     = 21;
const int VIEWER_RIGHT_EYE_OFFSET      = 22;
const int VIEWER_LEFT_EYE_NEAR_FAR     = 23;
const int VIEWER_MIDDLE_EYE_NEAR_FAR   = 24;
const int VIEWER_RIGHT_EYE_NEAR_FAR    = 25;
const int VIEWER_EYE_RECEIVER          = 26;

const int WINDOW_INIT_COMMAND          = 30;
const int WINDOW_EXIT_COMMAND          = 31;

const int SURFACE_ANGLE                = 40;
const int SURFACE_ROTCENTER            = 41;
const int SURFACE_ROTAXIS              = 42;
const int SURFACE_SIZE                 = 43;
const int SURFACE_UPVECTOR             = 44;
const int SURFACE_MATRIX               = 45;

const int CONTROLLER_TYPE              = 50;
const int CONTROLLER_SHARED_ARENA      = 51;
const int CONTROLLER_RECEIVER          = 52;
const int CONTROLLER_RECEIVER_OFFSET   = 53;

namespace SCIRun {

inline void tokerr( char* file, char* token, int line )
{
  cerr << "Error: Unknown token " << token << 
    " in configuration file " << file << " at line " << line << "." << endl;
}

inline void daterr( char* file, int line )
{
  cerr << "Error: Parsing data in configuration file " << file << 
    " at line " << line << "." << endl;
}



int BaWGL :: parseData( ifstream& f, char* fname, int& l, int t, int cnum, int rnum )
{
  char token[TOKENSIZE];
  int i, j;

  switch(t)
    {
    case TRACKER_TYPE:
      f >> token;
      if( f.fail() )
	{
	  daterr(fname, l);
	  return(-1);
	}
      if( !strcmp(token, "FASTRAK") )
	{ tracker.type = TRACKER_FASTRAK; }
      else if( !strcmp(token, "FOB") )
	{ tracker.type = TRACKER_FOB; }
      else
	{
	  cerr << "Error: Unknown tracker type " << token << " in configuration file "
	       << fname << " at line " << l << "." << endl;
	  return(-1);
	}
      break;
    case TRACKER_SHARED_ARENA:
      f >> tracker.arena;
      break;
    case TRACKER_TRANSMITTER_MATRIX:
      for( i=0; i<4; i++ )
	{
	  for( j=0; j<4; j++ )
	    {
	      f >> transmitterMatrix[i+4*j];
	      if( f.fail() )
		{
		  daterr(fname, l);
		  return(-1);
		}
	    }
	  l++;
	}
      l--;
      break;

    case VIEWER_LEFT_EYE_OFFSET:
      f >> eyeOffset[BAWGL_LEFT_EYE][0] >> eyeOffset[BAWGL_LEFT_EYE][1] >> eyeOffset[BAWGL_LEFT_EYE][2];
      break;
    case VIEWER_MIDDLE_EYE_OFFSET:
      f >> eyeOffset[BAWGL_MIDDLE_EYE][0] >> eyeOffset[BAWGL_MIDDLE_EYE][1] >> eyeOffset[BAWGL_MIDDLE_EYE][2];
      break;
    case VIEWER_RIGHT_EYE_OFFSET:
      f >> eyeOffset[BAWGL_RIGHT_EYE][0] >> eyeOffset[BAWGL_RIGHT_EYE][1] >> eyeOffset[BAWGL_RIGHT_EYE][2];
      break;

    case VIEWER_LEFT_EYE_NEAR_FAR:
      f >> nearClip[BAWGL_LEFT_EYE] >> farClip[BAWGL_LEFT_EYE];
      break;
    case VIEWER_MIDDLE_EYE_NEAR_FAR:
      f >> nearClip[BAWGL_MIDDLE_EYE] >> farClip[BAWGL_MIDDLE_EYE];
      break; 
    case VIEWER_RIGHT_EYE_NEAR_FAR:
      f >> nearClip[BAWGL_RIGHT_EYE] >> farClip[BAWGL_RIGHT_EYE];
      break;
    case VIEWER_EYE_RECEIVER:
      f >> eyeReceiver;
      break;

    case WINDOW_INIT_COMMAND:
      f >> windowInitCmd;
      i = strlen(windowInitCmd);
      f.getline(&windowInitCmd[i], CMDLEN-i);
      break;
    case WINDOW_EXIT_COMMAND:
      f >> windowExitCmd;
      i = strlen(windowExitCmd);
      f.getline(&windowExitCmd[i], CMDLEN-i);
      break;

    case SURFACE_ANGLE:
      f >> surfaceAngle;
      break;
    case SURFACE_ROTCENTER:
      f >> rotCenter[0] >> rotCenter[1] >> rotCenter[2];
      break;
    case SURFACE_ROTAXIS:
      f >> rotAxis[0] >> rotAxis[1] >> rotAxis[2];
      break;
    case SURFACE_SIZE:
      f >> surfaceWidth >> surfaceHeight;
      break;
    case SURFACE_UPVECTOR:
      f >> upVector[0] >> upVector[1] >> upVector[2];
      break;
    case SURFACE_MATRIX:
      for( i=0; i<4; i++ )
	{
	  for( j=0; j<4; j++ )
	    {
	      f >> surfaceMatrix[i+4*j];
	      if( f.fail() )
		{
		  daterr(fname, l);
		  return(-1);
		}
	    }
	  l++;
	}
      l--;
      haveSurfaceMatrix = true;
      break;

    case CONTROLLER_TYPE:
      f >> token;
      if( f.fail() )
	{
	  daterr(fname, l);
	  return(-1);
	}
      if( !strcmp(token, "PINCH") )
	{ controller[cnum].type = CONTROLLER_PINCH; }
      else if( !strcmp(token, "STYLUS") )
	{ controller[cnum].type = CONTROLLER_STYLUS; }
      else if( !strcmp(token, "I3STICK") )
	{ controller[cnum].type = CONTROLLER_I3STICK; }
      else if( !strcmp(token, "NONE") )
	{ controller[cnum].type = CONTROLLER_NONE; }
      else
	{
	  cerr << "Error: Unknown controller type " << token << 
	    " in configuration file " << fname << " at line " << l << "." << endl;
	  return(-1);
	}
      break;
    case CONTROLLER_SHARED_ARENA:
      f >> controller[cnum].arena;
      break;    
    case CONTROLLER_RECEIVER:
      f >> controller[cnum].receiver[rnum];
      break;
    case CONTROLLER_RECEIVER_OFFSET:
      for( i=0; i<4; i++ )
	{
	  for( j=0; j<4; j++ )
	    {
	      f >> controller[cnum].offset[rnum][i+4*j];
	      if( f.fail() )
		{
		  daterr(fname, l);
		  return(-1);
		}
	    }
	  l++;
	}
      l--;
      break;
    default:
      break;
    }
  
  if( f.fail() )
    {
      daterr(fname, l);
      return(-1);
    }
  l++;
  return(0);
}


int BaWGL :: parse( char* fname )
{
  char token1[TOKENSIZE], token2[TOKENSIZE];
  int l = 0, cnum, rnum;
  
  haveSurfaceMatrix = false;

  ifstream f(fname, ios::in);

  if( !f )
    {
      cerr << "Error: Cannot open configuration file " << fname << "." << endl;
      return(-1);
    }

  while( f >> token1 )
    {
      if( token1[0] != '#' )
        {  
	  f >> token2;

	  if( token1 && token2 )
	    {
	      if( !strcmp(token1, "TRACKER") )
		{
		  if( !strcmp(token2, "TYPE") )
		    { if( parseData(f, fname, l, TRACKER_TYPE) < 0 ) return(-1); }
		  else if( !strcmp(token2, "TRANSMITTER_MATRIX") )
		    { if( parseData(f, fname, l, TRACKER_TRANSMITTER_MATRIX) < 0 ) return(-1); }
		  else if( !strcmp(token2, "SHARED_ARENA") )
		    { if( parseData(f, fname, l, TRACKER_SHARED_ARENA) < 0 ) return(-1); }
		  else
		    {
		      tokerr(fname, token2, l);
		      return(-1);
		    }
		}
	      else if( !strcmp(token1, "VIEWER") )
		{
		  if( !strcmp(token2, "LEFT_EYE_OFFSET") )
		    { if( parseData(f, fname, l, VIEWER_LEFT_EYE_OFFSET) < 0 ) return(-1); }
		  else if( !strcmp(token2, "MIDDLE_EYE_OFFSET") )
		    { if( parseData(f, fname, l, VIEWER_MIDDLE_EYE_OFFSET) < 0 ) return(-1); }
		  else if( !strcmp(token2, "RIGHT_EYE_OFFSET") )
		    { if( parseData(f, fname, l, VIEWER_RIGHT_EYE_OFFSET) < 0 ) return(-1); }
		  else if( !strcmp(token2, "LEFT_EYE_NEAR_FAR") )
		    { if( parseData(f, fname, l, VIEWER_LEFT_EYE_NEAR_FAR) < 0 ) return(-1); }
		  else if( !strcmp(token2, "MIDDLE_EYE_NEAR_FAR") )
		    { if( parseData(f, fname, l, VIEWER_MIDDLE_EYE_NEAR_FAR) < 0 ) return(-1); }
		  else if( !strcmp(token2, "RIGHT_EYE_NEAR_FAR") )
		    { if( parseData(f, fname, l, VIEWER_RIGHT_EYE_NEAR_FAR) < 0 ) return(-1); }
		  else if( !strcmp(token2, "EYE_RECEIVER") )
		    { if( parseData(f, fname, l, VIEWER_EYE_RECEIVER) < 0 ) return(-1); }
		  else
		    {
		      tokerr(fname, token2, l);
		      return(-1);
		    }
		}
	      else if( !strcmp(token1, "WINDOW") )
		{
		  if( !strcmp(token2, "INIT_COMMAND") )
		    { if( parseData(f, fname, l, WINDOW_INIT_COMMAND) < 0 ) return(-1); }
		  else if( !strcmp(token2, "EXIT_COMMAND") )
		    { if( parseData(f, fname, l, WINDOW_EXIT_COMMAND) < 0 ) return(-1); }
		  else
		    {
		      tokerr(fname, token2, l);
		      return(-1);
		    }
		}
	      else if( !strcmp(token1, "SURFACE") )
		{
		  if( !strcmp(token2, "ANGLE") )
		    { if( parseData(f, fname, l, SURFACE_ANGLE) < 0 ) return(-1); }
		  else if( !strcmp(token2, "ROTCENTER") )
		    { if( parseData(f, fname, l, SURFACE_ROTCENTER) < 0 ) return(-1); }
		  else if( !strcmp(token2, "ROTAXIS") )
		    { if( parseData(f, fname, l, SURFACE_ROTAXIS) < 0 ) return(-1); }
		  else if( !strcmp(token2, "SIZE") )
		    { if( parseData(f, fname, l, SURFACE_SIZE) < 0 ) return(-1); }
		  else if( !strcmp(token2, "UPVECTOR") )
		    { if( parseData(f, fname, l, SURFACE_UPVECTOR) < 0 ) return(-1); }
		  else if( !strcmp(token2, "MATRIX") )
		    { if( parseData(f, fname, l, SURFACE_MATRIX) < 0 ) return(-1); }
		  else
		    {
		      tokerr(fname, token2, l);
		      return(-1);
		    }
		}
	      else if( sscanf(token1, "CONTROLLER%d", &cnum) == 1 )
		{
		  if( !strcmp(token2, "TYPE") )
		    { if( parseData(f, fname, l, CONTROLLER_TYPE, cnum) < 0 ) return(-1); }
		  else if( !strcmp(token2, "SHARED_ARENA") )
		    { if( parseData(f, fname, l, CONTROLLER_SHARED_ARENA, cnum) < 0 ) return(-1); }
		  else if( !strcmp(token2, "RECEIVER") )
		    { if( parseData(f, fname, l, CONTROLLER_RECEIVER, cnum) < 0 ) return(-1); }
		  else if( !strcmp(token2, "RECEIVER_OFFSET") )
		    { if( parseData(f, fname, l, CONTROLLER_RECEIVER_OFFSET, cnum) < 0 ) return(-1); }
		  else if( sscanf(token2, "RECEIVER%d", &rnum) == 1 )
		    { if( parseData(f, fname, l, CONTROLLER_RECEIVER, cnum, rnum) < 0 ) return(-1); }
		  else if( sscanf(token2, "RECEIVER_OFFSET%d", &rnum) == 1 )
		    { if( parseData(f, fname, l, CONTROLLER_RECEIVER_OFFSET, cnum, rnum) < 0 ) return(-1); }
		  else
		    {
		      tokerr(fname, token2, l);
		      return(-1);
		    }
		}
	      else
		{
		  tokerr(fname, token1, l);
		  return(-1);
		}
	    }
	  else
	    {
	      cerr << "Error: Parsing configuration file " << fname << 
		" at line " << l << "." << endl;
	      return(-1);
	    }
	}
      else
        { 
	  l++; 
	  f.getline(token2, TOKENSIZE);
	}
    }

  return(0);
}

} // End namespace SCIRun
