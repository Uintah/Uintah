//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : OSXOpenGLContext.h
//    Author : McKay Davis
//    Date   : Thu Sep  7 09:04:39 2006

#ifndef CORE_OSXOPENGLCONTEXT_H
#define CORE_OSXOPENGLCONTEXT_H

#if defined(__APPLE__)

#include <Core/Geom/OpenGLContext.h>
#include <Core/Thread/Mutex.h>

#include <Carbon/Carbon.h>
#include <AGL/agl.h>

namespace SCIRun {

class SCISHARE OSXOpenGLContext : public OpenGLContext 
{
public:
  OSXOpenGLContext(int x = 0,
                   int y = 0,
                   unsigned int width = 640, 
                   unsigned int height = 480,
                   bool border = true);

  virtual ~OSXOpenGLContext();

  virtual bool		make_current();
  virtual void		release();
  virtual int		width();
  virtual int		height();
  virtual void		swap();

  WindowPtr             window_;
private:  
  void                  create_context(int w, int h, 
                                       unsigned int width, 
                                       unsigned int height,
                                       bool border);
  
  AGLContext            context_;
  static AGLContext     first_context_;
  
  Mutex                 mutex_;
  int                   width_;
  int                   height_;
};

} // End namespace SCIRun

#endif

#endif // SCIRun_Core_2d_OpenGLContext_h
