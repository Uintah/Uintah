
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
 *  BundleGetColorMap.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleGetColorMap : public Module {
public:
  BundleGetColorMap(GuiContext*);

  virtual ~BundleGetColorMap();

  virtual void execute();
  
private:
  GuiString             guicolormap1name_;
  GuiString             guicolormap2name_;
  GuiString             guicolormap3name_;
  GuiString             guicolormaps_;
};


DECLARE_MAKER(BundleGetColorMap)
  BundleGetColorMap::BundleGetColorMap(GuiContext* ctx)
    : Module("BundleGetColorMap", ctx, Filter, "Bundle", "SCIRun"),
      guicolormap1name_(get_ctx()->subVar("colormap1-name"), "colormap1"),
      guicolormap2name_(get_ctx()->subVar("colormap2-name"), "colormap2"),
      guicolormap3name_(get_ctx()->subVar("colormap3-name"), "colormap3"),
      guicolormaps_(get_ctx()->subVar("colormap-selection"), "")
{
}

BundleGetColorMap::~BundleGetColorMap(){
}

void BundleGetColorMap::execute()
{
  string colormap1name = guicolormap1name_.get();
  string colormap2name = guicolormap2name_.get();
  string colormap3name = guicolormap3name_.get();
  string colormaplist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  ColorMapHandle fhandle;
  ColorMapOPort *ofport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
  {
    error("Could not find bundle input port");
    return;
  }

  if (!(iport->get(handle)))
  {   
    warning("No bundle connected to the input port");
    return;
  }

  if (handle.get_rep() == 0)
  {   
    warning("Empty bundle connected to the input port");
    return;
  }


  int numColorMaps = handle->numColorMaps();
  for (int p = 0; p < numColorMaps; p++)
  {
    colormaplist += "{" + handle->getColorMapName(p) + "} ";
  }

  guicolormaps_.set(colormaplist);
  get_ctx()->reset();

  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap1"))))
  {
    error("Could not find colormap 1 output port");
    return; 
  }
 
  if (handle->isColorMap(colormap1name))
  {
    fhandle = handle->getColorMap(colormap1name);
    ofport->send(fhandle);
  }
         
  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap2"))))
  {
    error("Could not find colormap 2 output port");
    return; 
  }
  
  if (handle->isColorMap(colormap2name))
  {
    fhandle = handle->getColorMap(colormap2name);
    ofport->send(fhandle);
  }
      
  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap3"))))
  {
    error("Could not find colormap 3 output port");
    return; 
  }

  if (handle->isColorMap(colormap3name))
  {
    fhandle = handle->getColorMap(colormap3name);
    ofport->send(fhandle);
  }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    oport->send(handle);
  }        
  update_state(Completed);  
}

