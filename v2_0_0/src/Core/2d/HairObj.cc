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
 *  HairObj.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Malloc/Allocator.h>
#include <Core/2d/HairObj.h>
#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>


namespace SCIRun {

Persistent* make_HairObj()
{
  return scinew HairObj;
}

PersistentTypeID HairObj::type_id("HairObj", "Widget", make_HairObj);

HairObj::HairObj( const string &name)
  : Widget(name), pos_(0.5)
{
}


HairObj::~HairObj()
{
}

void
HairObj::recompute()
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
HairObj::select( double , double , int  )
{
  recompute();
}
  
void
HairObj::move( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    pos_ = px;
}
  
void
HairObj::release( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    pos_ = px;
}
  

#define HAIRLINE_VERSION 1

void 
HairObj::io(Piostream& stream)
{
  stream.begin_class("HairObj", HAIRLINE_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
