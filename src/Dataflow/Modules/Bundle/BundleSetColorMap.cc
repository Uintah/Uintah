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
 *  BundleSetColorMap.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetColorMap : public Module {
public:
  BundleSetColorMap(GuiContext*);

  virtual ~BundleSetColorMap();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guicolormap1name_;
  GuiString     guicolormap2name_;
  GuiString     guicolormap3name_;
  GuiInt        guicolormap1usename_;
  GuiInt        guicolormap2usename_;
  GuiInt        guicolormap3usename_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetColorMap)
  BundleSetColorMap::BundleSetColorMap(GuiContext* ctx)
    : Module("BundleSetColorMap", ctx, Source, "Bundle", "SCIRun"),
      guicolormap1name_(ctx->subVar("colormap1-name")),
      guicolormap2name_(ctx->subVar("colormap2-name")),
      guicolormap3name_(ctx->subVar("colormap3-name")),
      guicolormap1usename_(ctx->subVar("colormap1-usename")),
      guicolormap2usename_(ctx->subVar("colormap2-usename")),
      guicolormap3usename_(ctx->subVar("colormap3-usename")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetColorMap::~BundleSetColorMap(){
}

void
BundleSetColorMap::execute()
{
  string colormap1name = guicolormap1name_.get();
  string colormap2name = guicolormap2name_.get();
  string colormap3name = guicolormap3name_.get();
  //int colormap1usename = guicolormap1usename_.get();
  //int colormap2usename = guicolormap2usename_.get();
  //int colormap3usename = guicolormap3usename_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  ColorMapHandle fhandle;
  ColorMapIPort *ifport;
        
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
  if (!(ifport = static_cast<ColorMapIPort *>(get_iport("colormap1"))))
    {
      error("Could not find colormap 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      /*  COMMENTED OUT UNTIL ColorMap has an option to set a name
          if (colormap1usename)
          {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
          {
          colormap1name = name;
          guicolormap1name_.set(name);
          ctx->reset();
          }
          }
      */
      handle->setColormap(colormap1name,fhandle);
    }

  // Scan colormap input port 2   
  if (!(ifport = static_cast<ColorMapIPort *>(get_iport("colormap2"))))
    {
      error("Could not find colormap 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      /*  COMMENTED OUT UNTIL ColorMap has an option to set a nam
          if (colormap2usename)
          {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
          {    
          colormap2name = name;
          guicolormap2name_.set(name);
          ctx->reset();
          }    
          }
      */        

      handle->setColormap(colormap2name,fhandle);
    }

  // Scan colormap input port 3   
  if (!(ifport = static_cast<ColorMapIPort *>(get_iport("colormap3"))))
    {
      error("Could not find colormap 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {

      /*  COMMENTED OUT UNTIL ColorMap has an option to set a name
          if (colormap3usename)
          {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
          {
          colormap3name = name;
          guicolormap1name_.set(name);
          ctx->reset();
          }
          }
      */
    
      handle->setColormap(colormap3name,fhandle);
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
        
  oport->send(handle);
}

void
BundleSetColorMap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




