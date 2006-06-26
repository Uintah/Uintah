//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : X11OpenGLContext.h
//    Author : McKay Davis
//    Date   : May 30 2006


#ifndef CORE_X11OPENGLCONTEXT_H
#define CORE_X11OPENGLCONTEXT_H

#include <sci_glx.h>

#ifdef __sgi
#  include <X11/extensions/SGIStereo.h>
#endif

#include <Core/Thread/Mutex.h>
#include <Core/Geom/OpenGLContext.h>

// X11 includes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xmu/StdCmap.h>


namespace SCIRun {

class SCISHARE X11OpenGLContext : public OpenGLContext 
{
public:
  X11OpenGLContext(int visualid=0, 
                   int x = 0,
                   int y = 0,
                   unsigned int width = 640, 
                   unsigned int height = 480);

  virtual ~X11OpenGLContext();
private:  
  void                  create_context(int id, int w, int h, 
                                       unsigned int width, 
                                       unsigned int height);

  static void		listvisuals();

public:
  virtual bool		make_current();
  virtual void		release();
  virtual int		width();
  virtual int		height();
  virtual void		swap();

private:
  static vector<int>	valid_visuals_;  
  static GLXContext     first_context_;

public:
  Display *		display_; /* X's token for the window's display. */
  int                   screen_;
  Window		window_;
private:
  Window		root_window_;
  int                   visualid_;

  GLXContext		context_;
  XVisualInfo*		vi_;
  Colormap		colormap_;
  XSetWindowAttributes  attributes_;
  Mutex                 mutex_;
  int                   width_;
  int                   height_;
};

} // End namespace SCIRun

#endif // SCIRun_Core_2d_OpenGLContext_h
