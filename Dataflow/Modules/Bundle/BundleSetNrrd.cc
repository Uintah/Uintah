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
 *  BundleSetNrrd.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetNrrd : public Module {
public:
  BundleSetNrrd(GuiContext*);

  virtual ~BundleSetNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guinrrd1name_;
  GuiString     guinrrd2name_;
  GuiString     guinrrd3name_;
  GuiInt        guinrrd1usename_;
  GuiInt        guinrrd2usename_;
  GuiInt        guinrrd3usename_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetNrrd)
  BundleSetNrrd::BundleSetNrrd(GuiContext* ctx)
    : Module("BundleSetNrrd", ctx, Source, "Bundle", "SCIRun"),
      guinrrd1name_(ctx->subVar("nrrd1-name")),
      guinrrd2name_(ctx->subVar("nrrd2-name")),
      guinrrd3name_(ctx->subVar("nrrd3-name")),
      guinrrd1usename_(ctx->subVar("nrrd1-usename")),
      guinrrd2usename_(ctx->subVar("nrrd2-usename")),
      guinrrd3usename_(ctx->subVar("nrrd3-usename")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetNrrd::~BundleSetNrrd(){
}

void
BundleSetNrrd::execute()
{
  string nrrd1name = guinrrd1name_.get();
  string nrrd2name = guinrrd2name_.get();
  string nrrd3name = guinrrd3name_.get();
  int nrrd1usename = guinrrd1usename_.get();
  int nrrd2usename = guinrrd2usename_.get();
  int nrrd3usename = guinrrd3usename_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  NrrdDataHandle fhandle;
  NrrdIPort *ifport;
        
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
  if (!(ifport = static_cast<NrrdIPort *>(get_iport("nrrd1"))))
    {
      error("Could not find nrrd 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd1usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              nrrd1name = name;
              guinrrd1name_.set(name);
              ctx->reset();
            }
        }
      handle->setNrrd(nrrd1name,fhandle);
    }

  // Scan nrrd input port 2     
  if (!(ifport = static_cast<NrrdIPort *>(get_iport("nrrd2"))))
    {
      error("Could not find nrrd 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd2usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {    
              nrrd2name = name;
              guinrrd2name_.set(name);
              ctx->reset();
            }    
        }

      handle->setNrrd(nrrd2name,fhandle);
    }

  // Scan nrrd input port 3     
  if (!(ifport = static_cast<NrrdIPort *>(get_iport("nrrd3"))))
    {
      error("Could not find nrrd 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd3usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              nrrd3name = name;
              guinrrd1name_.set(name);
              ctx->reset();
            }
        }
    
      handle->setNrrd(nrrd3name,fhandle);
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
BundleSetNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




