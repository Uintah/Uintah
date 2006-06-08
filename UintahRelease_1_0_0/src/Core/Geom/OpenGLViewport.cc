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
 *  OpenGLViewport.cc:
 *
 *  Written by:
 *   McKay Davis
 *   August 2004
 *
 */

#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/TkOpenGLContext.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>


#include <iostream>
using namespace SCIRun;
using namespace std;

OpenGLViewport::OpenGLViewport(TkOpenGLContext *ctx, 
			       float x, float y, float w, float h)
  : context_(ctx), // defaults to 0
    x_(x), // defaults to 0
    y_(y), // defaults to 0
    width_(w), // defaults to -1
    height_(h), // defaults to -1
    current_level_(0)
{
  check_bounds();
}


void
OpenGLViewport::check_bounds() {
  if (width_ < 0.0) 
    width_ = 1.0;
  if (height_ < 0.0)
    height_ = 1.0;
  
  if (x_ < 0.0 || x_ > 1.0)
    x_ = 0.0;

  if (y_ < 0.0 || y_ > 1.0)
    y_ = 0.0;
}


void
OpenGLViewport::resize(float x, float y, float w, float h) {
  x_ = x;
  y_ = y;
  width_ = w;
  height_ = h;
  check_bounds();
}

int
OpenGLViewport::max_width() {
  if (context_)
    return context_->width();
  else return 0;
}

int
OpenGLViewport::max_height() {
  if (context_)
    return context_->height();
  else return 0;
}

int
OpenGLViewport::x() {
  return Round(x_ * max_width());
}

int
OpenGLViewport::y() {
  return Round(y_ * max_height());
}

int
OpenGLViewport::width() {
  return Round(width_ * max_width());
}

int
OpenGLViewport::height() {
  return Round(height_ * max_height());
}


// just draw an opaque rectangle over area
// need to extend to back and other buffers
void 
OpenGLViewport::clear(float r, float g, float b, float a)
{
  if (!make_current()) return;
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glDrawBuffer(GL_BACK);
  glDisable(GL_BLEND);
  glColor4f(r,g,b,a);
  glBegin(GL_QUADS);
  glVertex3f(-1., -1., 0.);
  glVertex3f(1., -1., 0.);
  glVertex3f(1., 1., 0.);
  glVertex3f(-1., 1., 0.);
  glEnd();
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  release();
}
    
bool
OpenGLViewport::make_current()
{
  if (!current_level_++) {
    if (!context_->make_current()) {
      current_level_--;
      return false;
    }
  }

  check_bounds();
  glViewport(Round(x_*max_width()), Round(y_*max_height()), 
	     Round(width_*max_width()), Round(height_*max_height()));
  return true;
}


void
OpenGLViewport::swap()
{
  context_->swap();
}


void
OpenGLViewport::release()
{
  if (!--current_level_) {
    context_->release();
  }
}

