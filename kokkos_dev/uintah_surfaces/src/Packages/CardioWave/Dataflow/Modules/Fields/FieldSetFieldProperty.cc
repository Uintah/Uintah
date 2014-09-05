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
 *  FieldSetFieldProperty.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Core/Datatypes/Field.h>
#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class FieldSetFieldProperty : public Module {
public:
  FieldSetFieldProperty(GuiContext*);

  virtual ~FieldSetFieldProperty();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guifield1name_;
  GuiString     guifield2name_;
  GuiString     guifield3name_;
  GuiInt        guifield1usename_;
  GuiInt        guifield2usename_;
  GuiInt        guifield3usename_;
};


DECLARE_MAKER(FieldSetFieldProperty)
  FieldSetFieldProperty::FieldSetFieldProperty(GuiContext* ctx)
    : Module("FieldSetFieldProperty", ctx, Source, "Fields", "CardioWave"),
      guifield1name_(ctx->subVar("field1-name")),
      guifield2name_(ctx->subVar("field2-name")),
      guifield3name_(ctx->subVar("field3-name")),
      guifield1usename_(ctx->subVar("field1-usename")),
      guifield2usename_(ctx->subVar("field2-usename")),
      guifield3usename_(ctx->subVar("field3-usename"))
{
}

FieldSetFieldProperty::~FieldSetFieldProperty(){
}

void
FieldSetFieldProperty::execute()
{
  string field1name = guifield1name_.get();
  string field2name = guifield2name_.get();
  string field3name = guifield3name_.get();
  int field1usename = guifield1usename_.get();
  int field2usename = guifield2usename_.get();
  int field3usename = guifield3usename_.get();
    
  FieldHandle handle;
  FieldIPort  *iport;
  FieldOPort *oport;
  FieldHandle fhandle;
  FieldIPort *ifport;
        
  if(!(iport = static_cast<FieldIPort *>(get_iport("field"))))
    {
      error("Could not find field input port");
      return;
    }
      
  if (!(iport->get(handle)))
  {   
    error("Could not retrieve field from input port");
  }
   
  if (handle.get_rep() == 0)
  {
    error("No field on input port");
  }
  
  // Scan field input port 1
  if (!(ifport = static_cast<FieldIPort *>(get_iport("field1"))))
    {
      error("Could not find field 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (field1usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              field1name = name;
              guifield1name_.set(name);
              ctx->reset();
            }
        }
      handle->set_property(field1name,fhandle,false);
    }

  // Scan field input port 2     
  if (!(ifport = static_cast<FieldIPort *>(get_iport("field2"))))
    {
      error("Could not find field 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (field2usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {    
              field2name = name;
              guifield2name_.set(name);
              ctx->reset();
            }    
        }

      handle->set_property(field2name,fhandle,false);
    }

  // Scan field input port 3     
  if (!(ifport = static_cast<FieldIPort *>(get_iport("field3"))))
    {
      error("Could not find field 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (field3usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              field3name = name;
              guifield1name_.set(name);
              ctx->reset();
            }
        }
    
      handle->set_property(field3name,fhandle,false);
    }
        
  // Now post the output
        
  if (!(oport = static_cast<FieldOPort *>(get_oport("field"))))
    {
      error("Could not find field output port");
      return;
    }

  handle->generation++;                    
  oport->send(handle);
}

void
FieldSetFieldProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




