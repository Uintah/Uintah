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
 *  FieldSetStringProperty.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;
using namespace std;

class FieldSetStringProperty : public Module {
public:
  FieldSetStringProperty(GuiContext*);
  virtual ~FieldSetStringProperty();
  virtual void execute();
  
private:
  GuiString     guistring1name_;
  GuiString     guistring2name_;
  GuiString     guistring3name_;
};

DECLARE_MAKER(FieldSetStringProperty)
  FieldSetStringProperty::FieldSetStringProperty(GuiContext* ctx)
    : Module("FieldSetStringProperty", ctx, Source, "FieldsProperty", "ModelCreation"),
      guistring1name_(get_ctx()->subVar("string1-name")),
      guistring2name_(get_ctx()->subVar("string2-name")),
      guistring3name_(get_ctx()->subVar("string3-name"))
{
}

FieldSetStringProperty::~FieldSetStringProperty()
{
}

void
FieldSetStringProperty::execute()
{
  string string1name = guistring1name_.get();
  string string2name = guistring2name_.get();
  string string3name = guistring3name_.get();
    
  FieldHandle  handle;
  if (!get_input_handle("Field", handle)) return;
  
  // Scan field input port 1
  StringHandle fhandle;
  StringIPort  *ifport;
  if (!(ifport = static_cast<StringIPort *>(get_input_port("String1"))))
  {
    error("Could not find String 1 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    string fstring = fhandle->get();
    handle->set_property(string1name, fstring, false);
  }

  // Scan field input port 2     
  if (!(ifport = static_cast<StringIPort *>(get_input_port("String2"))))
    {
      error("Could not find String 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
  {
    string fstring = fhandle->get();  
    handle->set_property(string2name, fstring, false);
  }

  // Scan field input port 3     
  if (!(ifport = static_cast<StringIPort *>(get_input_port("String3"))))
    {
      error("Could not find String 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
  {
    string fstring = fhandle->get();  
    handle->set_property(string3name, fstring, false);
  }
        
  // Now post the output
  handle->generation++;
  send_output_handle("Field", handle);
}

} //end namespace
