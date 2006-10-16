
/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/ColorMap2Port.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleGetColorMap2 : public Module {
public:
  BundleGetColorMap2(GuiContext*);
  virtual void execute();
  
private:
  GuiString             guicolormap21name_;
  GuiString             guicolormap22name_;
  GuiString             guicolormap23name_;
  GuiString             guicolormap2s_;
};


DECLARE_MAKER(BundleGetColorMap2)

BundleGetColorMap2::BundleGetColorMap2(GuiContext* ctx)
  : Module("BundleGetColorMap2", ctx, Filter, "Bundle", "SCIRun"),
    guicolormap21name_(get_ctx()->subVar("colormap21-name"), "colormap21"),
    guicolormap22name_(get_ctx()->subVar("colormap22-name"), "colormap22"),
    guicolormap23name_(get_ctx()->subVar("colormap23-name"), "colormap23"),
    guicolormap2s_(get_ctx()->subVar("colormap2-selection"), "")
{
}

void
BundleGetColorMap2::execute()
{
  // Define input handle:
  BundleHandle handle;
  
  // Get data from input port:
  if (!(get_input_handle("bundle",handle,true))) return;
  
  if (inputs_changed_ || guicolormap21name_.changed() || guicolormap22name_.changed() ||
      guicolormap23name_.changed() || !oport_cached("bundle") || !oport_cached("colormap21") ||
       !oport_cached("colormap22") || !oport_cached("colormap23"))
  {
    ColorMapHandle fhandle;
    std::string colormap21name = guicolormap21name_.get();
    std::string colormap22name = guicolormap22name_.get();
    std::string colormap23name = guicolormap23name_.get();
    std::string colormap2list;
    
    int numColorMap2s = handle->numColorMap2s();
    for (int p = 0; p < numColorMap2s; p++)
    {
      colormap2list += "{" + handle->getColorMap2Name(p) + "} ";
    }

    guicolormap2s_.set(colormap2list);
    get_ctx()->reset();  
  
    // Send colormap1 if we found one that matches the name:
    if (handle->isColorMap(colormap21name))
    {
      fhandle = handle->getColorMap(colormap21name);
      send_output_handle("colormap21",fhandle,false);
    } 

    // Send colormap2 if we found one that matches the name:
    if (handle->isColorMap(colormap22name))
    {
      fhandle = handle->getColorMap(colormap22name);
      send_output_handle("colormap22",fhandle,false);
    } 

    // Send colormap3 if we found one that matches the name:
    if (handle->isColorMap(colormap23name))
    {
      fhandle = handle->getColorMap(colormap23name);
      send_output_handle("colormap23",fhandle,false);
    }
    
    send_output_handle("bundle",handle,false);     
  }
}

