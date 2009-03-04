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
//    File   : NrrdToColorMap2.cc
//    Author : Martin Cole
//    Date   : Fri Sep 24 09:48:56 2004

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/ColorMap2Port.h>
#include <Core/Volume/CM2Widget.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <iostream>

namespace SCIRun {

class NrrdToColorMap2 : public Module {

public:
  NrrdToColorMap2(GuiContext* ctx);
  virtual ~NrrdToColorMap2();

  virtual void execute();

private:
  ColorMap2Handle       ocmap_h_;
  int                   in_nrrd_gen_;
};



DECLARE_MAKER(NrrdToColorMap2)

NrrdToColorMap2::NrrdToColorMap2(GuiContext* ctx) : 
  Module("NrrdToColorMap2", ctx, Filter, "Converters", "Teem"),
  ocmap_h_(0),
  in_nrrd_gen_(-1)
{
}


NrrdToColorMap2::~NrrdToColorMap2()
{
}


void
NrrdToColorMap2::execute()
{
  NrrdDataHandle h;
  if (!get_input_handle("Image", h)) return;

  if (h->generation != in_nrrd_gen_ || !ocmap_h_.get_rep())
  {
    ocmap_h_ = 0;
    in_nrrd_gen_ = h->generation;
    if(h->nrrd_->dim != 3) {
      error("Invalid input dimension. Must be 3D.");
      return;
    }
    if (h->nrrd_->dim != 3 || h->nrrd_->axis[0].size != 4) {
      error("Invalid input size. Must be 4xWidthxHeigh.");
      return;
    }

    if (h->nrrd_->type != nrrdTypeFloat) {
      error("Input nrrd must be of type float: use UnuConvert.");
      return;
    }

    NrrdDataHandle temp = scinew NrrdData;
    nrrdFlip(temp->nrrd_, h->nrrd_, 2);

    vector<CM2WidgetHandle> widget;
    widget.push_back(scinew ImageCM2Widget(temp));

    ocmap_h_ = scinew ColorMap2(widget, false, -1, make_pair(0.0, -1.0));
  }

  send_output_handle("Output ColorMap", ocmap_h_, true);
}



} // end namespace SCIRun
