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
 *  AxesObj.cc: 
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Malloc/Allocator.h>
#include <Core/2d/AxesObj.h>
#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>


namespace SCIRun {

Persistent* make_XAxisObj()
{
  return scinew XAxisObj;
}

PersistentTypeID XAxisObj::type_id("XAxisObj", "Widget", make_XAxisObj);

XAxisObj::~XAxisObj()
{
}

void
XAxisObj::recompute()
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho( 0, 1,  0, 1,  -1, 1 );
  glGetDoublev( GL_PROJECTION_MATRIX, proj );
  glGetDoublev( GL_MODELVIEW_MATRIX, model );
  glGetIntegerv( GL_VIEWPORT, viewport );
  glPopMatrix();
}

void
XAxisObj::select( double , double , int  )
{
  recompute();
}
  
void
XAxisObj::move( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( py >= 0 && py < 1 )
    pos_ = py;
}
  
void
XAxisObj::release( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( py >= 0 && py < 1 )
    pos_ = py;
}

#define X_AXIS_VERSION 1

void 
XAxisObj::io(Piostream& stream)
{
  stream.begin_class("XAxisObj", X_AXIS_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}

Persistent* make_YAxisObj()
{
  return scinew YAxisObj;
}

PersistentTypeID YAxisObj::type_id("YAxisObj", "Widget", make_YAxisObj);

YAxisObj::~YAxisObj()
{
}

void
YAxisObj::recompute()
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho( 0, 1,  0, 1,  -1, 1 );
  glGetDoublev( GL_PROJECTION_MATRIX, proj );
  glGetDoublev( GL_MODELVIEW_MATRIX, model );
  glGetIntegerv( GL_VIEWPORT, viewport );
  glPopMatrix();
}

void
YAxisObj::select( double , double , int  )
{
  recompute();
}
  
void
YAxisObj::move( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    pos_ = px;
}
  
void
YAxisObj::release( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    pos_ = px;
}

#define Y_AXIS_VERSION 1

void 
YAxisObj::io(Piostream& stream)
{
  stream.begin_class("YAxisObj", Y_AXIS_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
