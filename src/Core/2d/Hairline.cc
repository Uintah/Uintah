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
 *  Hairline.cc: 
 *
 *  Written by:
 *   Yarden Livnat
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
#include <Core/2d/Hairline.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>


namespace SCIRun {

Persistent* make_Hairline()
{
  return scinew Hairline;
}

PersistentTypeID Hairline::type_id("Hairline", "Widget", make_Hairline);

Hairline::Hairline( const BBox2d &bbox, const string &name)
  : Widget(name), 
    from_(bbox.min().x()), to_(bbox.max().x()), pos_( (from_+to_)/2 )
{
}


Hairline::~Hairline()
{
}

void
Hairline::select( double x, double y, int  )
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho( from_, to_,  0, 1,  -1, 1 );
  glGetDoublev( GL_PROJECTION_MATRIX, proj );
  glGetDoublev( GL_MODELVIEW_MATRIX, model );
  glGetIntegerv( GL_VIEWPORT, viewport );
  glPopMatrix();
}
  
void
Hairline::move( double x, double y, int )
{
  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= from_ && px <= to_ )
    pos_ = px;
}
  
void
Hairline::release( double x, double y, int )
{
  GLdouble px, py, pz;
  gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
  if ( px >= from_ && px <= to_ )
    pos_ = px;
}
  

#define HAIRLINE_VERSION 1

void 
Hairline::io(Piostream& stream)
{
  stream.begin_class("Hairline", HAIRLINE_VERSION);
  Drawable::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
