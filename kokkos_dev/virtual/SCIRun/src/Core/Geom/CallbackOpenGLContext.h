//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : CallbackOpenGLContext.h
//    Author : Martin Cole
//    Date   : Fri May 26 14:36:23 2006

#if !defined(CallbackOpenGLContext_h)
#define CallbackOpenGLContext_h

#include <Core/Geom/OpenGLContext.h>

namespace SCIRun {

typedef int (*ZEROPARAMFUNC)(void*);

class SCISHARE CallbackOpenGLContext : public OpenGLContext 
{
public:

  CallbackOpenGLContext();
  virtual ~CallbackOpenGLContext();
  
  virtual bool			make_current();
  virtual void			release();
  virtual int			width();
  virtual int			height();
  virtual void			swap();

  void set_make_current_func(ZEROPARAMFUNC func, void *clientdata);
  void set_release_func(ZEROPARAMFUNC func, void *clientdata);
  void set_width_func(ZEROPARAMFUNC func, void *clientdata);
  void set_height_func(ZEROPARAMFUNC func, void *clientdata);
  void set_swap_func(ZEROPARAMFUNC func, void *clientdata);


private:
  ZEROPARAMFUNC                 make_current_func_;
  void*                         mc_cdata_;

  ZEROPARAMFUNC                 release_func_;
  void*                         r_cdata_;

  ZEROPARAMFUNC                 width_func_;
  void*                         w_cdata_;

  ZEROPARAMFUNC                 height_func_;
  void*                         h_cdata_;

  ZEROPARAMFUNC                 swap_func_;
  void*                         s_cdata_;
  
};

} // End namespace SCIRun

#endif // SCIRun_Core_2d_OpenGLContext_h
