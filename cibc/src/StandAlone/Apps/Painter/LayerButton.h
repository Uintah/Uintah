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
//    File   : LayerButton.h
//    Author : McKay Davis
//    Date   : Fri Oct 13 16:23:24 2006


#ifndef LEXOV_LayerButton
#define LEXOV_LayerButton

#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Variables.h>

namespace SCIRun {

class Painter;
class NrrdVolume;

class LayerButton : public Skinner::Parent {
public:
  LayerButton(Skinner::Variables *, Painter *painter);
  virtual ~LayerButton();
  CatcherFunction_t           update_from_gui;
private:
  friend class Painter;
  Painter *                   painter_;
  Skinner::Var<string>        layer_name_;
  Skinner::Var<int>           num_;
  Skinner::Var<double>        indent_;
  Skinner::Var<Skinner::Color>background_color_;
  Skinner::Var<bool>          layer_visible_;
  Skinner::Var<bool>          expand_;
  Skinner::Var<double>        expand_width_;
  
  NrrdVolumeHandle            volume_;
  CatcherFunction_t           up;
  CatcherFunction_t           down;
  CatcherFunction_t           kill;
  CatcherFunction_t           select;
  CatcherFunction_t           merge;
};


}

#endif
  
