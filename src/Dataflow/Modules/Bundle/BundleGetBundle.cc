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
 *  BundleGetBundle.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleGetBundle : public Module {
public:
  BundleGetBundle(GuiContext*);

  virtual ~BundleGetBundle();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guibundle1name_;
  GuiString             guibundle2name_;
  GuiString             guibundle3name_;
  GuiString             guibundles_;
};


DECLARE_MAKER(BundleGetBundle)
  BundleGetBundle::BundleGetBundle(GuiContext* ctx)
    : Module("BundleGetBundle", ctx, Source, "Bundle", "SCIRun"),
      guibundle1name_(ctx->subVar("bundle1-name")),
      guibundle2name_(ctx->subVar("bundle2-name")),
      guibundle3name_(ctx->subVar("bundle3-name")),
      guibundles_(ctx->subVar("bundle-selection"))
{

}


BundleGetBundle::~BundleGetBundle(){
}


void BundleGetBundle::execute()
{
  string bundle1name = guibundle1name_.get();
  string bundle2name = guibundle2name_.get();
  string bundle3name = guibundle3name_.get();
  string bundlelist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  BundleHandle fhandle;
  BundleOPort *ofport;
        
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


  int numBundles = handle->numBundles();
  for (int p = 0; p < numBundles; p++)
    {
      bundlelist += "{" + handle->getBundleName(p) + "} ";
    }

  guibundles_.set(bundlelist);
  ctx->reset();

 
  if (!(ofport = static_cast<BundleOPort *>(get_oport("bundle1"))))
  {
    error("Could not find bundle 1 output port");
    return; 
  }
 
  if (handle->isBundle(bundle1name))
  {
    fhandle = handle->getBundle(bundle1name);
    ofport->send(fhandle);
  }
        
 
  if (!(ofport = static_cast<BundleOPort *>(get_oport("bundle2"))))
  {
    error("Could not find bundle 2 output port");
    return; 
  }

 
  if (handle->isBundle(bundle2name))
  {
    fhandle = handle->getBundle(bundle2name);
    ofport->send(fhandle);
  }
        
 
  if (!(ofport = static_cast<BundleOPort *>(get_oport("bundle3"))))
  {
    error("Could not find bundle 3 output port");
    return; 
  }
 
  if (handle->isBundle(bundle3name))
  {
    fhandle = handle->getBundle(bundle3name);
    ofport->send(fhandle);
  }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    oport->send(handle);
  }
      
}

void BundleGetBundle::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

