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
 *  BundleSetPath.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Geom/Path.h>
#include <Dataflow/Ports/PathPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetPath : public Module {
public:
  BundleSetPath(GuiContext*);

  virtual ~BundleSetPath();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guipath1name_;
  GuiString     guipath2name_;
  GuiString     guipath3name_;
  GuiInt        guipath1usename_;
  GuiInt        guipath2usename_;
  GuiInt        guipath3usename_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetPath)
  BundleSetPath::BundleSetPath(GuiContext* ctx)
    : Module("BundleSetPath", ctx, Source, "Bundle", "SCIRun"),
      guipath1name_(ctx->subVar("path1-name")),
      guipath2name_(ctx->subVar("path2-name")),
      guipath3name_(ctx->subVar("path3-name")),
      guipath1usename_(ctx->subVar("path1-usename")),
      guipath2usename_(ctx->subVar("path2-usename")),
      guipath3usename_(ctx->subVar("path3-usename")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetPath::~BundleSetPath(){
}

void
BundleSetPath::execute()
{
  string path1name = guipath1name_.get();
  string path2name = guipath2name_.get();
  string path3name = guipath3name_.get();
  //int path1usename = guipath1usename_.get();
  //int path2usename = guipath2usename_.get();
  //int path3usename = guipath3usename_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  PathHandle fhandle;
  PathIPort *ifport;
        
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
  if (!(ifport = static_cast<PathIPort *>(get_iport("path1"))))
    {
      error("Could not find path 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      /*  COMMENTED OUT UNTIL Path has an option to set a name
          if (path1usename)
          {
          string name;
          handle->get_property("name",name);
          if (name != "") 
          {
          path1name = name;
          guipath1name_.set(name);
          ctx->reset();
          }
          }
      */
      handle->setPath(path1name,fhandle);
    }

  // Scan path input port 2       
  if (!(ifport = static_cast<PathIPort *>(get_iport("path2"))))
    {
      error("Could not find path 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      /*  COMMENTED OUT UNTIL Path has an option to set a name
          if (path2usename)
          {
          string name;
          handle->get_property("name",name);
          if (name != "") 
          {    
          path2name = name;
          guipath2name_.set(name);
          ctx->reset();
          }    
          }
      */
      handle->setPath(path2name,fhandle);
    }

  // Scan path input port 3       
  if (!(ifport = static_cast<PathIPort *>(get_iport("path3"))))
    {
      error("Could not find path 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    { 
    
      /*  COMMENTED OUT UNTIL ColorMap has an option to set a name
          if (path3usename)
          {
          string name;
          handle->get_property("name",name);
          if (name != "") 
          {
          path3name = name;
          guipath1name_.set(name);
          ctx->reset();
          }
          }
      */    
      handle->setPath(path3name,fhandle);
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
BundleSetPath::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




