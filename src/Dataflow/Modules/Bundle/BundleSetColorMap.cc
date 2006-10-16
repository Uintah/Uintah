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
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleSetColorMap : public Module {
public:
  BundleSetColorMap(GuiContext*);
  virtual void execute();

private:
  GuiString     guicolormap1name_;
  GuiString     guicolormap2name_;
  GuiString     guicolormap3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetColorMap)

BundleSetColorMap::BundleSetColorMap(GuiContext* ctx)
  : Module("BundleSetColorMap", ctx, Filter, "Bundle", "SCIRun"),
    guicolormap1name_(get_ctx()->subVar("colormap1-name"), "colormap1"),
    guicolormap2name_(get_ctx()->subVar("colormap2-name"), "colormap2"),
    guicolormap3name_(get_ctx()->subVar("colormap3-name"), "colormap3"),
    guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void BundleSetColorMap::execute()
{
  BundleHandle  handle;
  ColorMapHandle colormap1, colormap2, colormap3;

  get_input_handle("bundle",handle,false);
  get_input_handle("colormap1",colormap1,false);
  get_input_handle("colormap2",colormap2,false);
  get_input_handle("colormap3",colormap3,false);
  
  if (inputs_changed_ || guicolormap1name_.changed() || guicolormap2name_.changed() ||
      guicolormap3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string colormap1name = guicolormap1name_.get();
    std::string colormap2name = guicolormap2name_.get();
    std::string colormap3name = guicolormap3name_.get();
    std::string bundlename = guibundlename_.get();

    if (handle.get_rep())
    {
      handle.detach();
    }
    else
    {
      handle = scinew Bundle();
      if (handle.get_rep() == 0)
      {
        error("Could not allocate new bundle");
        return;
      }
    }
                
    if (colormap1.get_rep()) handle->setColorMap(colormap1name,colormap1);
    if (colormap2.get_rep()) handle->setColorMap(colormap2name,colormap2);
    if (colormap3.get_rep()) handle->setColorMap(colormap3name,colormap3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }
}
