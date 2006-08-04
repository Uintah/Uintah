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
//    File   : CallbackOpenGLContext.cc
//    Author : Martin Cole
//    Date   : Fri May 26 15:39:51 2006

#include <Core/Geom/CallbackOpenGLContext.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

CallbackOpenGLContext::CallbackOpenGLContext() :
  make_current_func_(0),
  mc_cdata_(0),
  release_func_(0),
  r_cdata_(0),
  width_func_(0),
  w_cdata_(0),
  height_func_(0),
  h_cdata_(0),
  swap_func_(0),
  s_cdata_(0)
{
}

CallbackOpenGLContext::~CallbackOpenGLContext() 
{
}

bool			
CallbackOpenGLContext::make_current()
{
  ASSERT(make_current_func_ && mc_cdata_);
  return (*make_current_func_)(mc_cdata_);
}

void			
CallbackOpenGLContext::release()
{
  ASSERT(release_func_ && r_cdata_);
  (*release_func_)(r_cdata_);
}

int			
CallbackOpenGLContext::width()
{
  ASSERT(width_func_ && w_cdata_);
  (*width_func_)(w_cdata_);
}

int			
CallbackOpenGLContext::height()
{
  ASSERT(height_func_ && h_cdata_);
  return (*height_func_)(h_cdata_);
}

void			
CallbackOpenGLContext::swap()
{
  ASSERT(swap_func_ && s_cdata_);
  (*swap_func_)(s_cdata_);
}

void 
CallbackOpenGLContext::set_make_current_func(ZEROPARAMFUNC func, 
					     void *cdata)
{
  make_current_func_ = func;
  mc_cdata_ = cdata;
}

void 
CallbackOpenGLContext::set_release_func(ZEROPARAMFUNC func, void *cdata)
{
  release_func_ = func;
  r_cdata_ = cdata;
}

void 
CallbackOpenGLContext::set_width_func(ZEROPARAMFUNC func, void *cdata)
{
  width_func_ = func;
  w_cdata_ = cdata;
}

void 
CallbackOpenGLContext::set_height_func(ZEROPARAMFUNC func, void *cdata)
{
  height_func_ = func;
  h_cdata_ = cdata;
}

void 
CallbackOpenGLContext::set_swap_func(ZEROPARAMFUNC func, void *cdata)
{
  swap_func_ = func;
  s_cdata_ = cdata;
}

} // End namespace SCIRun
