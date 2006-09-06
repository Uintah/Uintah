
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

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/PathPort.h>
#include <Core/Geom/Path.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleGetPath : public Module {
public:
  BundleGetPath(GuiContext*);
  virtual void execute();
  
private:
  GuiString             guipath1name_;
  GuiString             guipath2name_;
  GuiString             guipath3name_;
  GuiString             guipaths_;
};


DECLARE_MAKER(BundleGetPath)
  BundleGetPath::BundleGetPath(GuiContext* ctx)
    : Module("BundleGetPath", ctx, Filter, "Bundle", "SCIRun"),
      guipath1name_(get_ctx()->subVar("path1-name"), "path1"),
      guipath2name_(get_ctx()->subVar("path2-name"), "path2"),
      guipath3name_(get_ctx()->subVar("path3-name"), "path3"),
      guipaths_(get_ctx()->subVar("path-selection"), "")
{
}

void BundleGetPath::execute()
{
  // Define input handle:
  BundleHandle handle;
  
  // Get data from input port:
  if (!(get_input_handle("bundle",handle,true))) return;
  
  if (inputs_changed_ || guipath1name_.changed() || guipath2name_.changed() ||
      guipath3name_.changed() || !oport_cached("bundle") || !oport_cached("path1") ||
       !oport_cached("path2") || !oport_cached("path3"))
  {
    PathHandle fhandle;
    std::string path1name = guipath1name_.get();
    std::string path2name = guipath2name_.get();
    std::string path3name = guipath3name_.get();
    std::string pathlist;
      
    int numPaths = handle->numPaths();
    for (int p = 0; p < numPaths; p++)
    {
      pathlist += "{" + handle->getPathName(p) + "} ";
    }

    guipaths_.set(pathlist);
    get_ctx()->reset();

    // Send path1 if we found one that matches the name:
    if (handle->isPath(path1name))
    {
      fhandle = handle->getPath(path1name);
      send_output_handle("path1",fhandle,false);
    } 

    // Send path2 if we found one that matches the name:
    if (handle->isPath(path2name))
    {
      fhandle = handle->getPath(path2name);
      send_output_handle("path2",fhandle,false);
    } 

    // Send path3 if we found one that matches the name:
    if (handle->isPath(path3name))
    {
      fhandle = handle->getPath(path3name);
      send_output_handle("path3",fhandle,false);
    } 
  }
}

