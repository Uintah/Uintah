//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Animation.h
//    Author : McKay Davis
//    Date   : Tue Jul  4 17:47:07 2006


#ifndef SKINNER_ANIMATION_H
#define SKINNER_ANIMATION_H

#include <Core/Skinner/Parent.h>

namespace SCIRun {
  namespace Skinner {
    class Animation : public Parent {
    public:
      Animation (Variables *);
      virtual ~Animation();
      virtual MinMax                    get_minmax(unsigned int);
      static string                     class_name() { return "Animation"; }
      static DrawableMakerFunc_t        maker;
      
    private:
      CatcherFunction_t                 AnimateHeight;
      CatcherFunction_t                 AnimateVariableDescending;
      CatcherFunction_t                 AnimateVariable;
      CatcherFunction_t                 AnimateVariableAscending;
      double                            variable_begin_;
      double                            variable_end_;
      int                               curvar_;

      double                            start_time_;
      double                            stop_time_;

      bool                              at_start_;
      bool                              ascending_;

      TimeThrottle *                    timer_;
    };
  }
}

#endif
