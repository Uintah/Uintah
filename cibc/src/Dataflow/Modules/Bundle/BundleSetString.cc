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
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleSetString : public Module {
public:
  BundleSetString(GuiContext*);
  virtual void execute();
  
private:
  GuiString     guistring1name_;
  GuiString     guistring2name_;
  GuiString     guistring3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetString)

BundleSetString::BundleSetString(GuiContext* ctx)
  : Module("BundleSetString", ctx, Filter, "Bundle", "SCIRun"),
    guistring1name_(get_ctx()->subVar("string1-name"), "string1"),
    guistring2name_(get_ctx()->subVar("string2-name"), "string2"),
    guistring3name_(get_ctx()->subVar("string3-name"), "string3"),
    guibundlename_(get_ctx()->subVar("bundlename"))
{
}

void BundleSetString::execute()
{
  BundleHandle  handle;
  StringHandle string1, string2, string3;

  get_input_handle("bundle",handle,false);
  get_input_handle("string1",string1,false);
  get_input_handle("string2",string2,false);
  get_input_handle("string3",string3,false);
  
  if (inputs_changed_ || guistring1name_.changed() || guistring2name_.changed() ||
      guistring3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string string1name = guistring1name_.get();
    std::string string2name = guistring2name_.get();
    std::string string3name = guistring3name_.get();
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
                
    if (string1.get_rep()) handle->setString(string1name,string1);
    if (string2.get_rep()) handle->setString(string2name,string2);
    if (string3.get_rep()) handle->setString(string3name,string3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }
}

