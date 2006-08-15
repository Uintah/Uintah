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
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleSetField : public Module {
public:
  BundleSetField(GuiContext*);
  virtual void execute();

private:
  GuiString     guifield1name_;
  GuiString     guifield2name_;
  GuiString     guifield3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetField)
  BundleSetField::BundleSetField(GuiContext* ctx)
    : Module("BundleSetField", ctx, Filter, "Bundle", "SCIRun"),
      guifield1name_(get_ctx()->subVar("field1-name"), "field1"),
      guifield2name_(get_ctx()->subVar("field2-name"), "field2"),
      guifield3name_(get_ctx()->subVar("field3-name"), "field3"),
      guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void BundleSetField::execute()
{
  BundleHandle  handle;
  FieldHandle field1, field2, field3;

  get_input_handle("bundle",handle,false);
  get_input_handle("field1",field1,false);
  get_input_handle("field2",field2,false);
  get_input_handle("field3",field3,false);
  
  if (inputs_changed_ || guifield1name_.changed() || guifield2name_.changed() ||
      guifield3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string field1name = guifield1name_.get();
    std::string field2name = guifield2name_.get();
    std::string field3name = guifield3name_.get();
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
                
    if (field1.get_rep()) handle->setField(field1name,field1);
    if (field2.get_rep()) handle->setField(field2name,field2);
    if (field3.get_rep()) handle->setField(field3name,field3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }
}

