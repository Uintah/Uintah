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
//    File   : Root.h
//    Author : McKay Davis
//    Date   : Fri Jun 30 22:06:40 2006

#ifndef SKINNER_ROOT_H
#define SKINNER_ROOT_H

#include <Core/Skinner/Parent.h>
#include <string>
using std::string;

namespace SCIRun {
  namespace Skinner {
    class Variables;
    class GLWindow;
    class Root : public Parent {
    public:
      Root(Variables *);
      virtual ~Root();
      void spawn_redraw_threads();
    private:
      CatcherFunction_t Arithmetic_Maker;
      CatcherFunction_t GLWindow_Maker;
      CatcherFunction_t GLWindow_Destructor;
      CatcherFunction_t ColorMap2D_Maker;
      CatcherFunction_t Graph2D_Maker;
      CatcherFunction_t Quit;
      CatcherFunction_t Redraw;
      typedef vector<GLWindow *> GLWindows_t;
      GLWindows_t windows_;
    };
  }
}



#endif
