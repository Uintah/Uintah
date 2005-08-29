
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
 *  BundleGetField.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleGetField : public Module {

public:
  BundleGetField(GuiContext*);

  virtual ~BundleGetField();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guifield1name_;
  GuiString             guifield2name_;
  GuiString             guifield3name_;
  GuiString             guifields_;
};


DECLARE_MAKER(BundleGetField)
  BundleGetField::BundleGetField(GuiContext* ctx)
    : Module("BundleGetField", ctx, Source, "Bundle", "SCIRun"),
      guifield1name_(ctx->subVar("field1-name")),
      guifield2name_(ctx->subVar("field2-name")),
      guifield3name_(ctx->subVar("field3-name")),
      guifields_(ctx->subVar("field-selection"))
{

}

BundleGetField::~BundleGetField(){
}


void BundleGetField::execute()
{
  string field1name = guifield1name_.get();
  string field2name = guifield2name_.get();
  string field3name = guifield3name_.get();
  string fieldlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  SCIRun::FieldHandle fhandle;
  SCIRun::FieldOPort *ofport;
        
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

  if (handle.get_rep() == 0)
  {   
    warning("Empty bundle connected to the input port");
    return;
  }

  int numfields = handle->numFields();
  for (int p = 0; p < numfields; p++)
  {
    fieldlist += "{" + handle->getFieldName(p) + "} ";
  }

  guifields_.set(fieldlist);
  ctx->reset();

 
  if (!(ofport = static_cast<FieldOPort *>(get_oport("field1"))))
  {
    error("Could not find field 1 output port");
    return; 
  }

  if (handle->isField(field1name))
  {
    fhandle = handle->getField(field1name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<FieldOPort *>(get_oport("field2"))))
  {
    error("Could not find field 2 output port");
    return; 
  }
 
  if (handle->isField(field2name))
  {
    fhandle = handle->getField(field2name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<FieldOPort *>(get_oport("field3"))))
  {
    error("Could not find field 3 output port");
    return; 
  }
 
  if (handle->isField(field3name))
  {
    fhandle = handle->getField(field3name);
    ofport->send(fhandle);
  }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    oport->send(handle);
  }
        
}

void BundleGetField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




