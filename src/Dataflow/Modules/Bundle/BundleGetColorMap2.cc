
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
 *  BundleGetColorMap2.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/Colormap2Port.h>
#include <Core/Volume/Colormap2.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class BundleGetColorMap2 : public Module {
public:
  BundleGetColorMap2(GuiContext*);

  virtual ~BundleGetColorMap2();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guicolormap21name_;
  GuiString             guicolormap22name_;
  GuiString             guicolormap23name_;
  GuiString             guicolormap2s_;
};


DECLARE_MAKER(BundleGetColorMap2)
  BundleGetColorMap2::BundleGetColorMap2(GuiContext* ctx)
    : Module("BundleGetColorMap2", ctx, Source, "Bundle", "SCIRun"),
      guicolormap21name_(ctx->subVar("colormap21-name")),
      guicolormap22name_(ctx->subVar("colormap22-name")),
      guicolormap23name_(ctx->subVar("colormap23-name")),
      guicolormap2s_(ctx->subVar("colormap2-selection"))
{
}

BundleGetColorMap2::~BundleGetColorMap2(){
}

void
BundleGetColorMap2::execute()
{
  string colormap21name = guicolormap21name_.get();
  string colormap22name = guicolormap22name_.get();
  string colormap23name = guicolormap23name_.get();
  string colormap2list;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  ColorMap2Handle fhandle;
  ColorMap2OPort *ofport;
        
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

  int numColormap2s = handle->numColormap2s();
  for (int p = 0; p < numColormap2s; p++)
    {
      colormap2list += "{" + handle->getColormap2Name(p) + "} ";
    }

  guicolormap2s_.set(colormap2list);
  ctx->reset();

 
  if (!(ofport = static_cast<ColorMap2OPort *>(get_oport("colormap21"))))
    {
      error("Could not find colormap2 1 output port");
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
          colormap21name = nrrdstring.getstring();
          guicolormap21name_.set(colormap21name);
          ctx->reset();
        }
    }

 
  if (handle->isColormap2(colormap21name))
    {
      fhandle = handle->getColormap2(colormap21name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<ColorMap2OPort *>(get_oport("colormap22"))))
    {
      error("Could not find colormap2 2 output port");
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
          colormap22name = nrrdstring.getstring();
          guicolormap22name_.set(colormap22name);
          ctx->reset();
        }
    }

 
  if (handle->isColormap2(colormap22name))
    {
      fhandle = handle->getColormap2(colormap22name);
      ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<ColorMap2OPort *>(get_oport("colormap23"))))
    {
      error("Could not find colormap2 3 output port");
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
          colormap23name = nrrdstring.getstring();
          guicolormap23name_.set(colormap23name);
          ctx->reset();
        }
    }

 
  if (handle->isColormap2(colormap23name))
    {
      fhandle = handle->getColormap2(colormap23name);
      ofport->send(fhandle);
    }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      oport->send(handle);
    }
        
}

void
BundleGetColorMap2::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

