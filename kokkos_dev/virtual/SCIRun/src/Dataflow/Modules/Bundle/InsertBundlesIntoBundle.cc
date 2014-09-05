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
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class InsertBundlesIntoBundle : public Module {
public:
  InsertBundlesIntoBundle(GuiContext*);
  virtual void execute();

private:
  GuiString     guiBundle1Name_;
  GuiString     guiBundle2Name_;
  GuiString     guiBundle3Name_;
  GuiString     guiBundleName_;
};


DECLARE_MAKER(InsertBundlesIntoBundle)

InsertBundlesIntoBundle::InsertBundlesIntoBundle(GuiContext* ctx)
  : Module("InsertBundlesIntoBundle", ctx, Filter, "Bundle", "SCIRun"),
    guiBundle1Name_(get_ctx()->subVar("bundle1-name"), "bundle1"),
    guiBundle2Name_(get_ctx()->subVar("bundle2-name"), "bundle2"),
    guiBundle3Name_(get_ctx()->subVar("bundle3-name"), "bundle3"),
    guiBundleName_(get_ctx()->subVar("bundlename"), "")
{
}


void InsertBundlesIntoBundle::execute()
{
  BundleHandle  handle, bundle1, bundle2, bundle3;

  get_input_handle("bundle",handle,false);
  get_input_handle("bundle1",bundle1,false);
  get_input_handle("bundle2",bundle2,false);
  get_input_handle("bundle3",bundle3,false);
  
  if (inputs_changed_ || guiBundle1Name_.changed() || guiBundle2Name_.changed() ||
      guiBundle3Name_.changed() || guiBundleName_.changed() || !oport_cached("bundle"))
  {
  
    std::string bundle1Name = guiBundle1Name_.get();
    std::string bundle2Name = guiBundle2Name_.get();
    std::string bundle3Name = guiBundle3Name_.get();
    std::string bundleName = guiBundleName_.get();

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
                
    if (bundle1.get_rep()) handle->setBundle(bundle1Name,bundle1);
    if (bundle2.get_rep()) handle->setBundle(bundle2Name,bundle2);
    if (bundle3.get_rep()) handle->setBundle(bundle3Name,bundle3);
    if (bundleName != "")
    {
      handle->set_property("name",bundleName,false);
    }

    send_output_handle("bundle",handle,false);
  }
}

