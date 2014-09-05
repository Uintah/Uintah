/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <sci_gl.h>
#include <sci_glu.h>

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

  
