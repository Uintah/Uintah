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
 *  OpenGLWindow.h:
 *
 *  Written by:
 *   McKay Davis
 *   August 2004
 */


#ifndef SCIRun_Core_2d_OpenGLContext_h
#define SCIRun_Core_2d_OpenGLContext_h

#include <Core/Geom/TkOpenGLContext.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>


using std::string;

namespace SCIRun {
class GuiInterface;

class OpenGLContext {
private:
  TkOpenGLContext *	tk_gl_context_;
  GuiInterface *	gui_;
  bool			get_client_data();
public:
  OpenGLContext(GuiInterface* gui, const string &);
  virtual ~OpenGLContext();
  bool		make_current(bool do_lock=true);
  void		swap(bool do_release=false);
  void		release();
  int		xres();
  int		yres();
  Display *	display() { return tk_gl_context_->display_; }
  int		screen_number() { return tk_gl_context_->screen_number_; }
  GuiInterface *gui() { return gui_; }
};

} // End namespace SCIRun

#endif // SCIRun_Core_2d_OpenGLContext_h


