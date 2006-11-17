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
//    File   : Animation.cc
//    Author : McKay Davis
//    Date   : Tue Jul  4 17:46:58 2006



#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Animation.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Timer.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Events/EventManager.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    Animation::Animation(Variables *variables) :
      Parent(variables),
      variable_begin_(0),
      variable_end_(0),
      start_time_(0),
      stop_time_(0),
      at_start_(true),
      ascending_(true),
      timer_(new TimeThrottle())
    {
      REGISTER_CATCHER_TARGET(Animation::AnimateVariableAscending);
      REGISTER_CATCHER_TARGET(Animation::AnimateVariableDescending);
    }

    Animation::~Animation()
    {
      delete timer_;
    }

    MinMax
    Animation::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    Drawable *
    Animation::maker(Variables *variables)
    {
      return new Animation(variables);
    }
    
    BaseTool::propagation_state_e
    Animation::AnimateVariableAscending(event_handle_t e) {
      if (!at_start_) return STOP_E;
      ascending_ = true;
      return AnimateVariable(e);
    }

    BaseTool::propagation_state_e
    Animation::AnimateVariableDescending(event_handle_t e) {
      if (at_start_) return STOP_E;
      ascending_ = false;
      return AnimateVariable(e);
    }
      
    BaseTool::propagation_state_e
    Animation::AnimateVariable(event_handle_t e) {
      Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(e.get_rep());
      ASSERT(signal);
      Var<int> varnum(signal->get_vars(), "varnum");
      curvar_ = varnum();

      const string prefix = "variable-"+to_string(curvar_)+"-";

      bool stopped = timer_->current_state() == Timer::Stopped;
      Var<double> seconds(get_vars(), prefix+"seconds");
      
      if (stopped) {
        timer_->start();
      }      

      start_time_ = timer_->time();
      stop_time_ = start_time_ + seconds();

      variable_begin_ = Var<double>(get_vars(), prefix+"begin")();
      variable_end_ = Var<double>(get_vars(), prefix+"end")();

      Var<double>variable(get_vars(), 
                          "Animation::variable-"+to_string(curvar_));

      while (timer_->current_state() == Timer::Running) 
      {
        double time = timer_->time();
        double dt = (time - start_time_) / (stop_time_ - start_time_);

        if (!ascending_) {
          dt = 1.0 - dt;
        }

        double newval = variable_begin_ + (variable_end_-variable_begin_) * dt;

        newval = Clamp(newval, 
                       Min(variable_begin_, variable_end_),
                       Max(variable_begin_, variable_end_));
        //        height = newval;
        variable = newval;

        cerr << newval << std::endl;
        timer_->wait_for_time(time+1/60.0);
        if (time > stop_time_) {
          at_start_ = !at_start_;
          timer_->stop();
          variable = ascending_ ? variable_end_ : variable_begin_;
          
        }
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      }
      
      return CONTINUE_E;
    }

  }
}
