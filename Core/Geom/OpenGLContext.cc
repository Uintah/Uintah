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
  
extern "C" Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

OpenGLContext::OpenGLContext(GuiInterface* gui, const string &id)
  : tk_clientdata_(0),
    gui_(gui),
    id_(id)
{
}

OpenGLContext::~OpenGLContext()
{
}

bool
OpenGLContext::get_client_data() {
  Tcl_CmdInfo info;
  if(!Tcl_GetCommandInfo(the_interp, const_cast<char *>(id_.c_str()), &info))
    return false;

  tk_clientdata_=(OpenGLClientData*)info.clientData;
  return tk_clientdata_;
}

  
bool
OpenGLContext::make_current(bool lock)
{
  if (lock) gui_->lock();

  if (!tk_clientdata_ && !get_client_data())
    return false;

  if (!tk_clientdata_->cx || !tk_clientdata_->glx_win)
    OpenGLGetContext(the_interp, const_cast<char *>(id_.c_str()));
  
  if (!tk_clientdata_->cx) {
    if (lock) gui_->unlock();
    throw InternalError("OpenGLContext unable to get GL context id: "+id_);
  }

  if (!tk_clientdata_->glx_win) {
    if (lock) gui_->unlock();
    throw InternalError("OpenGLContext unable to get GLX Window id: "+id_);
  }

#if 0
  if (!glXMakeContextCurrent(tk_clientdata_->display,
			     tk_clientdata_->glx_win,
			     tk_clientdata_->glx_win,
			     tk_clientdata_->cx)) 
#else
  if (!glXMakeCurrent(tk_clientdata_->display,
		      tk_clientdata_->x11_win,
		      tk_clientdata_->cx)) 
#endif
    
  {

    std::cerr << id_ << " failed make current.\n";
    if (lock) gui_->unlock();
    return false;
  }
  
  return true;
}


void
OpenGLContext::swap(bool do_release)
{
  if (!tk_clientdata_ && !get_client_data())
    return;

  ASSERT(tk_clientdata_->display);
  ASSERT(tk_clientdata_->glx_win);
  glXSwapBuffers(tk_clientdata_->display, tk_clientdata_->glx_win);
  if (do_release) release();
}


void
OpenGLContext::release()
{
  if (!tk_clientdata_ && !get_client_data())
    return;

  ASSERT(tk_clientdata_->display);
  glXMakeCurrent(tk_clientdata_->display, None, NULL);
}

int
OpenGLContext::xres() {
  if (!tk_clientdata_ && !get_client_data() || !tk_clientdata_->x11_win)
    return 0;

  ASSERT(tk_clientdata_);
  return (Tk_Width(tk_clientdata_->tkwin));
}

int
OpenGLContext::yres() {
  if (!tk_clientdata_ && !get_client_data() || !tk_clientdata_->x11_win)
    return 0;

  ASSERT(tk_clientdata_);
  return (Tk_Height(tk_clientdata_->tkwin));
}
