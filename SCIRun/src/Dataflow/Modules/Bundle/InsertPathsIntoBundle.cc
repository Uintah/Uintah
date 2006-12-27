/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <Core/Geom/Path.h>
#include <Dataflow/Network/Ports/PathPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class InsertPathsIntoBundle : public Module {
public:
  InsertPathsIntoBundle(GuiContext*);
  virtual void execute();
  
private:
  GuiString     guipath1name_;
  GuiString     guipath2name_;
  GuiString     guipath3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(InsertPathsIntoBundle)
  InsertPathsIntoBundle::InsertPathsIntoBundle(GuiContext* ctx)
    : Module("InsertPathsIntoBundle", ctx, Filter, "Bundle", "SCIRun"),
      guipath1name_(get_ctx()->subVar("path1-name"), "path1"),
      guipath2name_(get_ctx()->subVar("path2-name"), "path2"),
      guipath3name_(get_ctx()->subVar("path3-name"), "path3"),
      guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void InsertPathsIntoBundle::execute()
{
  BundleHandle  handle;
  PathHandle path1, path2, path3;

  get_input_handle("bundle",handle,false);
  get_input_handle("path1",path1,false);
  get_input_handle("path2",path2,false);
  get_input_handle("path3",path3,false);
  
  if (inputs_changed_ || guipath1name_.changed() || guipath2name_.changed() ||
      guipath3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string path1name = guipath1name_.get();
    std::string path2name = guipath2name_.get();
    std::string path3name = guipath3name_.get();
    std::string bundlename = guibundlename_.get();

    if (handle.get_rep())
    {
      handle.detach();
    }
    else
    {
      handle = scinew Bundle();
      if (handle.get_rep() == 0)
      {
        error("Could not allocate new bundle");
        return;
      }
    }
                
    if (path1.get_rep()) handle->setPath(path1name,path1);
    if (path2.get_rep()) handle->setPath(path2name,path2);
    if (path3.get_rep()) handle->setPath(path3name,path3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }
}


