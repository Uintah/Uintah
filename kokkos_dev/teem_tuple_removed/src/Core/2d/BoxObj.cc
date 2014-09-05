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
 *  BoxObj.cc: 
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

#include <Core/Malloc/Allocator.h>
#include <Core/2d/BoxObj.h>
#include <Core/2d/Point2d.h>
#include <Core/2d/Vector2d.h>
#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

using namespace std;

namespace SCIRun {

Persistent* make_BoxObj()
{
  return scinew BoxObj;
}

PersistentTypeID BoxObj::type_id("BoxObj", "Widget", make_BoxObj);

BoxObj::BoxObj( const string &name)
  : Widget(name), screen_(BBox2d(Point2d(0.4,0.4), Point2d(0.6,0.6)))
{
}


BoxObj::~BoxObj()
{
}

void
BoxObj::recompute( bool)
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho( 0, 1,  1, 0,  -1, 1 );
    
  glGetDoublev( GL_PROJECTION_MATRIX, proj );
  glGetDoublev( GL_MODELVIEW_MATRIX, model );
  glGetIntegerv( GL_VIEWPORT, viewport );

  glPopMatrix();
}

void
BoxObj::select( double x, double y, int button )
{
  GLdouble pz;
  
  mode_ = button;
  recompute();
  gluUnProject( x, y, 0, model, proj, viewport, &sx_, &sy_, &pz );
}
  
void
BoxObj::move( double x, double y, int)
{
  GLdouble px, py, pz;

  recompute();
  switch ( mode_ ) {
  case 1:
    {
      gluUnProject( x, y, 0, model, proj, viewport, &px, &py, &pz );
      Vector2d d( px-sx_, py-sy_ );
      screen_ = BBox2d( screen_.min() + d, screen_.max() + d );
      sx_ = px;
      sy_ = py;
    }
    break;
  case 2:
    cerr << "move button 2" << endl;
    break;
  case 3:
    cerr << "move button 3" << endl;
    break;
  }
}
  
void
BoxObj::release( double , double , int )
{
}
  

#define BOXLINE_VERSION 1

void 
BoxObj::io(Piostream& stream)
{
  stream.begin_class("BoxObj", BOXLINE_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
