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
//    File   : LayerButton.cc
//    Author : McKay Davis
//    Date   : Tue Sep 26 21:56:14 2006


#include <StandAlone/Apps/Painter/Painter.h>

namespace SCIRun {

LayerButton::LayerButton(Skinner::Variables *vars, Painter *painter) :
  Parent(vars),
  painter_(painter),
  layer_name_(vars, "LayerButton::name"),
  num_(vars, "LayerButton::num"),
  indent_(vars, "LayerButton::indent",0),
  background_color_(vars, "LayerButton::background_color"),
  volume_(0)
  
{
  REGISTER_CATCHER_TARGET(LayerButton::up);
  REGISTER_CATCHER_TARGET(LayerButton::down);
  REGISTER_CATCHER_TARGET(LayerButton::kill);
  REGISTER_CATCHER_TARGET(LayerButton::select);
}

LayerButton::~LayerButton() 
{}


BaseTool::propagation_state_e
LayerButton::down(event_handle_t signalh) {
  if (volume_)
    painter_->move_layer_down(volume_);
  return CONTINUE_E;
}

BaseTool::propagation_state_e
LayerButton::up(event_handle_t signalh) {
  if (volume_)
    painter_->move_layer_up(volume_);
  return CONTINUE_E;
}


BaseTool::propagation_state_e
LayerButton::kill(event_handle_t signalh) {
  return CONTINUE_E;
}

BaseTool::propagation_state_e
LayerButton::select(event_handle_t signalh) {
  painter_->current_volume_ = volume_;
  painter_->rebuild_layer_buttons();
  return CONTINUE_E;
}


}
