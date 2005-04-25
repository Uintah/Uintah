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
 *  BundleSetBundle.cc:
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

class BundleSetBundle : public Module {
public:
  BundleSetBundle(GuiContext*);

  virtual ~BundleSetBundle();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guiBundle1Name_;
  GuiString     guiBundle2Name_;
  GuiString     guiBundle3Name_;
  GuiInt        guiBundle1UseName_;
  GuiInt        guiBundle2UseName_;
  GuiInt        guiBundle3UseName_;
  GuiString     guiBundleName_;
};


DECLARE_MAKER(BundleSetBundle)
  BundleSetBundle::BundleSetBundle(GuiContext* ctx)
    : Module("BundleSetBundle", ctx, Source, "Bundle", "SCIRun"),
      guiBundle1Name_(ctx->subVar("bundle1-name")),
      guiBundle2Name_(ctx->subVar("bundle2-name")),
      guiBundle3Name_(ctx->subVar("bundle3-name")),
      guiBundle1UseName_(ctx->subVar("bundle1-usename")),
      guiBundle2UseName_(ctx->subVar("bundle2-usename")),
      guiBundle3UseName_(ctx->subVar("bundle3-usename")),
      guiBundleName_(ctx->subVar("bundlename"))
{
}

BundleSetBundle::~BundleSetBundle(){
}

void
BundleSetBundle::execute()
{
  string bundle1Name = guiBundle1Name_.get();
  string bundle2Name = guiBundle2Name_.get();
  string bundle3Name = guiBundle3Name_.get();
  int    bundle1UseName = guiBundle1UseName_.get();
  int    bundle2UseName = guiBundle2UseName_.get();
  int    bundle3UseName = guiBundle3UseName_.get();
  string bundleName = guiBundleName_.get();
    
  BundleHandle  handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort  *oport;
  BundleHandle  fhandle;
  BundleIPort  *ifport;
        
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
  if (!(ifport = static_cast<BundleIPort *>(get_iport("bundle1"))))
    {
      error("Could not find bundle 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (bundle1UseName)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              bundle1Name = name;
              guiBundle1Name_.set(name);
              ctx->reset();
            }
        }
      handle->setBundle(bundle1Name,fhandle);
    }

  // Scan bundle input port 2     
  if (!(ifport = static_cast<BundleIPort *>(get_iport("bundle2"))))
    {
      error("Could not find bundle 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (bundle2UseName)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {    
              bundle2Name = name;
              guiBundle2Name_.set(name);
              ctx->reset();
            }    
        }

      handle->setBundle(bundle2Name,fhandle);
    }

  // Scan bundle input port 3     
  if (!(ifport = static_cast<BundleIPort *>(get_iport("bundle3"))))
    {
      error("Could not find bundle 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (bundle3UseName)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              bundle3Name = name;
              guiBundle1Name_.set(name);
              ctx->reset();
            }
        }
    
      handle->setBundle(bundle3Name,fhandle);
    }
        
  // Now post the output
        
  if (!(oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      error("Could not find bundle output port");
      return;
    }
    
  if (bundleName != "")
    {
      handle->set_property("name",bundleName,false);
    }
        
  oport->send(handle);
}

void
BundleSetBundle::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
