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
 *  FieldGetNrrdPropertyProperty.cc:
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

class FieldGetNrrdProperty : public Module {
public:
  FieldGetNrrdProperty(GuiContext*);

  virtual ~FieldGetNrrdProperty();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guinrrd1name_;
  GuiString             guinrrd2name_;
  GuiString             guinrrd3name_;
  GuiString             guinrrds_;
};


DECLARE_MAKER(FieldGetNrrdProperty)
  FieldGetNrrdProperty::FieldGetNrrdProperty(GuiContext* ctx)
    : Module("FieldGetNrrdProperty", ctx, Source, "Fields", "CardioWave"),
      guinrrd1name_(ctx->subVar("nrrd1-name")),
      guinrrd2name_(ctx->subVar("nrrd2-name")),
      guinrrd3name_(ctx->subVar("nrrd3-name")),
      guinrrds_(ctx->subVar("nrrd-selection"))
{

}

FieldGetNrrdProperty::~FieldGetNrrdProperty(){
}


void
FieldGetNrrdProperty::execute()
{
  string nrrd1name = guinrrd1name_.get();
  string nrrd2name = guinrrd2name_.get();
  string nrrd3name = guinrrd3name_.get();
  string nrrdlist;
        
  FieldHandle handle;
  FieldIPort  *iport;
  NrrdOPort *ofport;
  NrrdDataHandle fhandle;
        
  if(!(iport = static_cast<FieldIPort *>(getIPort("field"))))
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
    if (fhandle.get_rep()) nrrdlist += "{" + handle->get_property_name(p) + "} ";
  }

  guinrrds_.set(nrrdlist);
  ctx->reset();
 
 
   if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd1"))))
    {
      error("Could not find nrrd 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(getIPort("name1"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
          NrrdString nrrdstring(nrrdH); 
          nrrd2name = nrrdstring.getstring();
          guinrrd1name_.set(nrrd1name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(nrrd1name))
    {
      handle->get_property(nrrd1name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd3"))))
    {
      error("Could not find nrrd 3 output port");
      return; 
    }

 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd2"))))
    {
      error("Could not find nrrd 2 output port");
      return; 
    }
 
   niport = static_cast<NrrdIPort *>(getIPort("name2"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
          NrrdString nrrdstring(nrrdH); 
          nrrd2name = nrrdstring.getstring();
          guinrrd2name_.set(nrrd2name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(nrrd2name))
    {
      handle->get_property(nrrd2name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<NrrdOPort *>(get_oport("nrrd3"))))
    {
      error("Could not find nrrd 3 output port");
      return; 
    }
 
  niport = static_cast<NrrdIPort *>(getIPort("name3"));
  if (niport)
    {
      NrrdDataHandle nrrdH;
      niport->get(nrrdH);
      if (nrrdH.get_rep() != 0)
        {
          NrrdString nrrdstring(nrrdH); 
          nrrd3name = nrrdstring.getstring();
          guinrrd3name_.set(nrrd3name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(nrrd3name))
    {
      handle->get_property(nrrd3name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }

        
}

void
FieldGetNrrdProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




