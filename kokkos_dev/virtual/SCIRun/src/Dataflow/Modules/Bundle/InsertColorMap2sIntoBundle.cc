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

class InsertColorMap2sIntoBundle : public Module {
public:
  InsertColorMap2sIntoBundle(GuiContext*);
  virtual void execute();

private:
  GuiString     guicolormap21name_;
  GuiString     guicolormap22name_;
  GuiString     guicolormap23name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(InsertColorMap2sIntoBundle)

InsertColorMap2sIntoBundle::InsertColorMap2sIntoBundle(GuiContext* ctx)
  : Module("InsertColorMap2sIntoBundle", ctx, Filter, "Bundle", "SCIRun"),
    guicolormap21name_(get_ctx()->subVar("colormap21-name"), "colormap21"),
    guicolormap22name_(get_ctx()->subVar("colormap22-name"), "colormap22"),
    guicolormap23name_(get_ctx()->subVar("colormap23-name"), "colormap23"),
    guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void InsertColorMap2sIntoBundle::execute()
{
  BundleHandle  handle;
  ColorMap2Handle colormap21, colormap22, colormap23;

  get_input_handle("bundle",handle,false);
  get_input_handle("colormap21",colormap21,false);
  get_input_handle("colormap22",colormap22,false);
  get_input_handle("colormap23",colormap23,false);
  
  if (inputs_changed_ || guicolormap21name_.changed() || guicolormap22name_.changed() ||
      guicolormap23name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string colormap21name = guicolormap21name_.get();
    std::string colormap22name = guicolormap22name_.get();
    std::string colormap23name = guicolormap23name_.get();
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
                
    if (colormap21.get_rep()) handle->setColorMap2(colormap21name,colormap21);
    if (colormap22.get_rep()) handle->setColorMap2(colormap22name,colormap22);
    if (colormap23.get_rep()) handle->setColorMap2(colormap23name,colormap23);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }
}


