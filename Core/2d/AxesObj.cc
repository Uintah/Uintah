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


#include <stdio.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <sstream>
using std::ostringstream;

#include <Core/Malloc/Allocator.h>
#include <Core/2d/AxesObj.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>


namespace SCIRun {

Persistent* make_AxesObj()
{
  return scinew AxesObj;
}

PersistentTypeID AxesObj::type_id("AxesObj", "Widget", make_AxesObj);

AxesObj::AxesObj( const string &name)
  : HairObj(name), xpos_(0.5), ypos_(0.5), num_h_tics_(7), num_v_tics_(5)
{
}


AxesObj::~AxesObj()
{
}

void
AxesObj::recompute()
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
AxesObj::select( double x, double y, int  )
{
  recompute();
}
  
void
AxesObj::move( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    xpos_ = px;
  if ( py >= 0 && py < 1 )
    ypos_ = py;
}
  
void
AxesObj::release( double x, double y, int )
{
  recompute();

  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= 0 && px < 1 )
    xpos_ = px;
  if ( py >= 0 && py < 1 )
    ypos_ = py;
}
  

#define TWO_D_AXES_VERSION 1

void 
AxesObj::io(Piostream& stream)
{
  stream.begin_class("AxesObj", TWO_D_AXES_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
