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
//    File   : TkOpenGLEventSpawner.h
//    Author : McKay Davis
//    Date   : Fri Jun  2 13:37:16 MDT 2006

#ifndef TKOPENGLEVENTSPAWNER_H
#define TKOPENGLEVENTSPAWNER_H


#include <Core/GuiInterface/GuiCallback.h>
#include <Core/Events/EventSpawner.h>

namespace SCIRun {
  class TkOpenGLContext;
  
  class TkOpenGLEventSpawner : public EventSpawner, public GuiCallback
  {
  public:
    TkOpenGLEventSpawner(TkOpenGLContext *);
    ~TkOpenGLEventSpawner();
    virtual void              tcl_command(GuiArgs &, void *);
  private:
    TkOpenGLContext *         context_;
  };
}

#endif
