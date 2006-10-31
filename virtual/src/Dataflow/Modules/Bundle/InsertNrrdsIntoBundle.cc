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
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class InsertNrrdsIntoBundle : public Module {
public:
  InsertNrrdsIntoBundle(GuiContext*);
  virtual void execute();

private:
  GuiString     guinrrd1name_;
  GuiString     guinrrd2name_;
  GuiString     guinrrd3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(InsertNrrdsIntoBundle)

InsertNrrdsIntoBundle::InsertNrrdsIntoBundle(GuiContext* ctx)
  : Module("InsertNrrdsIntoBundle", ctx, Filter, "Bundle", "SCIRun"),
    guinrrd1name_(get_ctx()->subVar("nrrd1-name"), "nrrd1"),
    guinrrd2name_(get_ctx()->subVar("nrrd2-name"), "nrrd2"),
    guinrrd3name_(get_ctx()->subVar("nrrd3-name"), "nrrd3"),
    guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void
InsertNrrdsIntoBundle::execute()
{
  BundleHandle  handle;
  NrrdDataHandle nrrd1, nrrd2, nrrd3;

  get_input_handle("bundle",handle,false);
  get_input_handle("nrrd1",nrrd1,false);
  get_input_handle("nrrd2",nrrd2,false);
  get_input_handle("nrrd3",nrrd3,false);
  
  if (inputs_changed_ || guinrrd1name_.changed() || guinrrd2name_.changed() ||
      guinrrd3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string nrrd1name = guinrrd1name_.get();
    std::string nrrd2name = guinrrd2name_.get();
    std::string nrrd3name = guinrrd3name_.get();
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
                
    if (nrrd1.get_rep()) handle->setNrrd(nrrd1name,nrrd1);
    if (nrrd2.get_rep()) handle->setNrrd(nrrd2name,nrrd2);
    if (nrrd3.get_rep()) handle->setNrrd(nrrd3name,nrrd3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }

}





