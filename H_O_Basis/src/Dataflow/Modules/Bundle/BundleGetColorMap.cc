
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
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class BundleGetColorMap : public Module {
public:
  BundleGetColorMap(GuiContext*);

  virtual ~BundleGetColorMap();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guicolormap1name_;
  GuiString             guicolormap2name_;
  GuiString             guicolormap3name_;
  GuiString             guicolormaps_;
};


DECLARE_MAKER(BundleGetColorMap)
  BundleGetColorMap::BundleGetColorMap(GuiContext* ctx)
    : Module("BundleGetColorMap", ctx, Source, "Bundle", "SCIRun"),
      guicolormap1name_(ctx->subVar("colormap1-name")),
      guicolormap2name_(ctx->subVar("colormap2-name")),
      guicolormap3name_(ctx->subVar("colormap3-name")),
      guicolormaps_(ctx->subVar("colormap-selection"))
{
}

BundleGetColorMap::~BundleGetColorMap(){
}

void
BundleGetColorMap::execute()
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


  int numColormaps = handle->numColormaps();
  for (int p = 0; p < numColormaps; p++)
    {
      colormaplist += "{" + handle->getColormapName(p) + "} ";
    }

  guicolormaps_.set(colormaplist);
  ctx->reset();

 
  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap1"))))
    {
      error("Could not find colormap 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(getIPort("name1"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          colormap1name = nrrdstring.getstring();
          guicolormap1name_.set(colormap1name);
          ctx->reset();
        }
    }
 
  if (handle->isColormap(colormap1name))
    {
      fhandle = handle->getColormap(colormap1name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap2"))))
    {
      error("Could not find colormap 2 output port");
      return; 
    }
 
  niport = static_cast<NrrdIPort *>(getIPort("name2"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          colormap2name = nrrdstring.getstring();
          guicolormap2name_.set(colormap2name);
          ctx->reset();
        }
    }

 
 
  if (handle->isColormap(colormap2name))
    {
      fhandle = handle->getColormap(colormap2name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<ColorMapOPort *>(get_oport("colormap3"))))
    {
      error("Could not find colormap 3 output port");
      return; 
    }

  niport = static_cast<NrrdIPort *>(getIPort("name3"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
    
          NrrdString nrrdstring(nrrdH); 
          colormap3name = nrrdstring.getstring();
          guicolormap3name_.set(colormap3name);
          ctx->reset();
        }
    }

  if (handle->isColormap(colormap3name))
    {
      fhandle = handle->getColormap(colormap3name);
      ofport->send(fhandle);
    }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      oport->send(handle);
    }
        
}

void
BundleGetColorMap::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


