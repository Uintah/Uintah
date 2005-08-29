
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
 *  BundleGetPath.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/PathPort.h>
#include <Core/Geom/Path.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleGetPath : public Module {
public:
  BundleGetPath(GuiContext*);

  virtual ~BundleGetPath();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guipath1name_;
  GuiString             guipath2name_;
  GuiString             guipath3name_;
  GuiString             guipaths_;
};


DECLARE_MAKER(BundleGetPath)
  BundleGetPath::BundleGetPath(GuiContext* ctx)
    : Module("BundleGetPath", ctx, Source, "Bundle", "SCIRun"),
      guipath1name_(ctx->subVar("path1-name")),
      guipath2name_(ctx->subVar("path2-name")),
      guipath3name_(ctx->subVar("path3-name")),
      guipaths_(ctx->subVar("path-selection"))
{

}

BundleGetPath::~BundleGetPath(){
}

void BundleGetPath::execute()
{
  string path1name = guipath1name_.get();
  string path2name = guipath2name_.get();
  string path3name = guipath3name_.get();
  string pathlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  PathHandle fhandle;
  PathOPort *ofport;
        
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

  int numPaths = handle->numPaths();
  for (int p = 0; p < numPaths; p++)
  {
    pathlist += "{" + handle->getPathName(p) + "} ";
  }


  if (handle.get_rep() == 0)
  {   
    warning("Empty bundle connected to the input port");
    return;
  }


  guipaths_.set(pathlist);
  ctx->reset();

 
  if (!(ofport = static_cast<PathOPort *>(get_oport("path1"))))
  {
    error("Could not find path 1 output port");
    return; 
  }
 
   if (handle->isPath(path1name))
  {
    fhandle = handle->getPath(path1name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<PathOPort *>(get_oport("path2"))))
  {
    error("Could not find path 2 output port");
    return; 
  }
 
  if (handle->isPath(path2name))
  {
    fhandle = handle->getPath(path2name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<PathOPort *>(get_oport("path3"))))
  {
    error("Could not find path 3 output port");
    return; 
  }
    
  if (handle->isPath(path3name))
  {
    fhandle = handle->getPath(path3name);
    ofport->send(fhandle);
  }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    oport->send(handle);
  }
        
}

void BundleGetPath::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


