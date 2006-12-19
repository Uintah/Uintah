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

/*
 *  InsertStringIntoFieldProperties.cc:
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

class InsertStringIntoFieldProperties : public Module {
public:
  InsertStringIntoFieldProperties(GuiContext*);
  virtual ~InsertStringIntoFieldProperties();
  virtual void execute();
  
private:
  GuiString     guistring1name_;
  GuiString     guistring2name_;
  GuiString     guistring3name_;
};

DECLARE_MAKER(InsertStringIntoFieldProperties)
  InsertStringIntoFieldProperties::InsertStringIntoFieldProperties(GuiContext* ctx)
    : Module("InsertStringIntoFieldProperties", ctx, Source, "FieldProperty", "ModelCreation"),
      guistring1name_(get_ctx()->subVar("string1-name")),
      guistring2name_(get_ctx()->subVar("string2-name")),
      guistring3name_(get_ctx()->subVar("string3-name"))
{
}

InsertStringIntoFieldProperties::~InsertStringIntoFieldProperties()
{
}

void
InsertStringIntoFieldProperties::execute()
{
  const string string1name = guistring1name_.get();
  const string string2name = guistring2name_.get();
  const string string3name = guistring3name_.get();
    
  FieldHandle handle;
  if (!get_input_handle("Field", handle)) return;
  
  // Scan field input port 1
  StringHandle shandle;
  if (get_input_handle("String1", shandle, false))
  {
    const string fstring = shandle->get();
    handle->set_property(string1name, fstring, false);
  }

  // Scan field input port 2
  if (get_input_handle("String2", shandle, false))
  {
    const string fstring = shandle->get();  
    handle->set_property(string2name, fstring, false);
  }

  // Scan field input port 3
  if (get_input_handle("String3", shandle, false))
  {
    const string fstring = shandle->get();  
    handle->set_property(string3name, fstring, false);
  }
        
  // Now post the output
  handle->generation = handle->compute_new_generation();
  send_output_handle("Field", handle);
}

} //end namespace
