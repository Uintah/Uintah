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
 *  OpenGLContext.cc:
 *
 *  Written by:
 *   McKay Davis
 *   August 2004
 *
 */


#include <Core/Geom/OpenGLContext.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Malloc/Allocator.h>
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
  

OpenGLContext::OpenGLContext(GuiInterface* gui, const string &id)
  : tk_gl_context_(scinew TkOpenGLContext(id, 640, 480, 53)),
    gui_(gui)
{
  ASSERT(tk_gl_context_);
}

OpenGLContext::~OpenGLContext()
{
  delete tk_gl_context_;
}

bool
OpenGLContext::make_current(bool lock)
{
  if (lock) gui_->lock();

  if (!tk_gl_context_->make_current()) 
  {
    if (lock) gui_->unlock();
    return false;
  }
  
  return true;
}


void
OpenGLContext::swap(bool do_release)
{
  tk_gl_context_->swap();
  if (do_release) tk_gl_context_->release();
}


void
OpenGLContext::release()
{
  tk_gl_context_->release();
}

int
OpenGLContext::xres() {
  return tk_gl_context_->width();
}

int
OpenGLContext::yres() {
  return tk_gl_context_->height();
}
