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
#include <sci_gl.h>
#include <sci_glu.h>


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

  
