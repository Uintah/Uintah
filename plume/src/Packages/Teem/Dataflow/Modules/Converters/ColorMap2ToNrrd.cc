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
//    File   : ColorMap2ToNrrd.cc
//    Author : Michael Callahan
//    Date   : Fri Oct 15 2004

#include <Dataflow/Network/Module.h>
#include <Core/Volume/Colormap2.h>
#include <Core/Volume/CM2Widget.h>
#include <Dataflow/Ports/Colormap2Port.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

namespace SCIRun {

class ColorMap2ToNrrd : public Module {

public:
  ColorMap2ToNrrd(GuiContext* ctx);
  virtual ~ColorMap2ToNrrd();

  virtual void execute();

private:
  int                   last_generation_;
};



DECLARE_MAKER(ColorMap2ToNrrd)

ColorMap2ToNrrd::ColorMap2ToNrrd(GuiContext* ctx) : 
  Module("ColorMap2ToNrrd", ctx, Filter, "Converters", "Teem"),
  last_generation_(-1)
{
}


ColorMap2ToNrrd::~ColorMap2ToNrrd()
{
}


void
ColorMap2ToNrrd::execute()
{
  ColorMap2IPort* cmap2_port = (ColorMap2IPort*)get_iport("Input");
  if (!cmap2_port)
  {
    error("Could not get input port 'Input'.");
    return;
  }
  
  ColorMap2Handle cmap2;
  if (!(cmap2_port->get(cmap2) && cmap2.get_rep()))
  {
    error("No data in input port.");
    return;
  }

  if (cmap2->generation != last_generation_)
  {
    last_generation_ = cmap2->generation;

    Array3<float> adata(256, 512, 4);
    adata.initialize(0.0);

    // Rasterize widgets
    vector<CM2WidgetHandle>& widget = cmap2->widgets();
    for (unsigned int i=0; i<widget.size(); i++)
      widget[i]->rasterize(adata);

    // Clean up any image overflows.
    for (int i=0; i<adata.dim1(); i++)
    {
      for (int j=0; j<adata.dim2(); j++)
      {
	adata(i,j,0) = Clamp(adata(i,j,0), 0.0f, 1.0f);
	adata(i,j,1) = Clamp(adata(i,j,1), 0.0f, 1.0f);
	adata(i,j,2) = Clamp(adata(i,j,2), 0.0f, 1.0f);
	adata(i,j,3) = Clamp(adata(i,j,3), 0.0f, 1.0f);
      }
    }

    // Create the nrrd.
    NrrdData *nd = scinew NrrdData();
    nrrdAlloc(nd->nrrd, nrrdTypeFloat, 3, 4, 512, 256);
    nd->nrrd->axis[2].kind = nrrdKindDomain;
    nd->nrrd->axis[2].label = airStrdup("Gradient Magnitude");
    nd->nrrd->axis[1].kind = nrrdKindDomain;
    nd->nrrd->axis[1].label = airStrdup("Data Value");
    nd->nrrd->axis[0].kind = nrrdKind4Color;
    nd->nrrd->axis[0].label = airStrdup("Colors");

    // Move the image to the nrrd.
    float *data = (float *)nd->nrrd->data;
    const int h = adata.dim1() - 1;
    for (int i=0; i<adata.dim1(); i++)
    {
      for (int j=0; j<adata.dim2(); j++)
      {
	*data++ = adata(h-i,j,0);
	*data++ = adata(h-i,j,1);
	*data++ = adata(h-i,j,2);
	*data++ = adata(h-i,j,3);
      }
    }

    NrrdDataHandle nrrd_h(nd);

    NrrdOPort* nrrd_port = (NrrdOPort*)get_oport("Output Image");
    if (!nrrd_port)
    {
      error("Could not open output port 'Output Image'.");
      return;
    }
    nrrd_port->send(nrrd_h);
  }
}



} // end namespace SCIRun
