/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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

/*
 *  BundleSetColorMap2.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/ColorMap2Port.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetColorMap2 : public Module {
public:
  BundleSetColorMap2(GuiContext*);

  virtual ~BundleSetColorMap2();
  virtual void execute();

private:
  GuiString     guicolormap21name_;
  GuiString     guicolormap22name_;
  GuiString     guicolormap23name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetColorMap2)

BundleSetColorMap2::BundleSetColorMap2(GuiContext* ctx)
  : Module("BundleSetColorMap2", ctx, Filter, "Bundle", "SCIRun"),
    guicolormap21name_(get_ctx()->subVar("colormap21-name"), "colormap21"),
    guicolormap22name_(get_ctx()->subVar("colormap22-name"), "colormap22"),
    guicolormap23name_(get_ctx()->subVar("colormap23-name"), "colormap23"),
    guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}


BundleSetColorMap2::~BundleSetColorMap2()
{
}


void
BundleSetColorMap2::execute()
{
  string colormap21name = guicolormap21name_.get();
  string colormap22name = guicolormap22name_.get();
  string colormap23name = guicolormap23name_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  ColorMap2Handle fhandle;
  ColorMap2IPort *ifport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
    {
      error("Could not find bundle input port");
      return;
    }
        
  // Create a new bundle
  // Since a bundle consists of only handles we can copy
  // it several times without too much memory overhead
  if (iport->get(oldhandle))
  {   // Copy all the handles from the existing bundle
    handle = oldhandle->clone();
  }
  else
  {   // Create a brand new bundle
    handle = scinew Bundle;
  }
        
  // Scan bundle input port 1
  if (!(ifport = static_cast<ColorMap2IPort *>(get_iport("colormap21"))))
  {
    error("Could not find colormap2 1 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->setColorMap2(colormap21name,fhandle);
  }

  // Scan colormap2 input port 2  
  if (!(ifport = static_cast<ColorMap2IPort *>(get_iport("colormap22"))))
  {
    error("Could not find colormap2 2 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->setColorMap2(colormap22name,fhandle);
  }

  // Scan colormap2 input port 3  
  if (!(ifport = static_cast<ColorMap2IPort *>(get_iport("colormap23"))))
  {
    error("Could not find colormap2 3 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->setColorMap2(colormap23name,fhandle);
  }
        
  // Now post the output
        
  if (!(oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    error("Could not find bundle output port");
    return;
  }
    
  if (bundlename != "")
  {
    handle->set_property("name",bundlename,false);
  }
        
  oport->send_and_dereference(handle);
  
  update_state(Completed);  
}


