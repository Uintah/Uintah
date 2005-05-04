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
 *  FieldGetFieldPropertyProperty.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class FieldGetFieldProperty : public Module {
public:
  FieldGetFieldProperty(GuiContext*);

  virtual ~FieldGetFieldProperty();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guifield1name_;
  GuiString             guifield2name_;
  GuiString             guifield3name_;
  GuiString             guifields_;
};


DECLARE_MAKER(FieldGetFieldProperty)
  FieldGetFieldProperty::FieldGetFieldProperty(GuiContext* ctx)
    : Module("FieldGetFieldProperty", ctx, Source, "Fields", "CardioWave"),
      guifield1name_(ctx->subVar("field1-name")),
      guifield2name_(ctx->subVar("field2-name")),
      guifield3name_(ctx->subVar("field3-name")),
      guifields_(ctx->subVar("field-selection"))
{

}

FieldGetFieldProperty::~FieldGetFieldProperty(){
}


void
FieldGetFieldProperty::execute()
{
  string field1name = guifield1name_.get();
  string field2name = guifield2name_.get();
  string field3name = guifield3name_.get();
  string fieldlist;
        
  FieldHandle handle;
  FieldIPort  *iport;
  FieldOPort *ofport;
  FieldHandle fhandle;
        
  if(!(iport = static_cast<FieldIPort *>(get_iport("field"))))
    {
      error("Could not find field input port");
      return;
    }

  if (!(iport->get(handle)))
    {   
      warning("No field connected to the input port");
      return;
    }

  if (handle.get_rep() == 0)
    {   
      warning("Empty field connected to the input port");
      return;
    }

  size_t nprop = handle->nproperties();

  for (size_t p=0;p<nprop;p++)
  {
    handle->get_property(handle->get_property_name(p),fhandle);
    if (fhandle.get_rep()) fieldlist += "{" + handle->get_property_name(p) + "} ";
  }

  guifields_.set(fieldlist);
  ctx->reset();
 
 
   if (!(ofport = static_cast<FieldOPort *>(get_oport("field1"))))
    {
      error("Could not find field 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(get_iport("name1"));
  if (niport)
    {
      NrrdDataHandle fieldH;
      niport->get(fieldH);
      if (fieldH.get_rep() != 0)
        {
          NrrdString fieldstring(fieldH); 
          field2name = fieldstring.getstring();
          guifield1name_.set(field1name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(field1name))
    {
      handle->get_property(field1name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }
 
  if (!(ofport = static_cast<FieldOPort *>(get_oport("field2"))))
    {
      error("Could not find field 2 output port");
      return; 
    }
 
   niport = static_cast<NrrdIPort *>(get_iport("name2"));
  if (niport)
    {
      NrrdDataHandle fieldH;
      niport->get(fieldH);
      if (fieldH.get_rep() != 0)
        {
          NrrdString fieldstring(fieldH); 
          field2name = fieldstring.getstring();
          guifield2name_.set(field2name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(field2name))
    {
      handle->get_property(field2name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<FieldOPort *>(get_oport("field3"))))
    {
      error("Could not find field 3 output port");
      return; 
    }
 
  niport = static_cast<NrrdIPort *>(get_iport("name3"));
  if (niport)
    {
      NrrdDataHandle fieldH;
      niport->get(fieldH);
      if (fieldH.get_rep() != 0)
        {
          NrrdString fieldstring(fieldH); 
          field3name = fieldstring.getstring();
          guifield3name_.set(field3name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(field3name))
    {
      handle->get_property(field3name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }

        
}

void
FieldGetFieldProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




