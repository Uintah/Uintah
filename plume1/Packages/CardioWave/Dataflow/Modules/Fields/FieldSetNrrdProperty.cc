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
 *  FieldSetNrrdProperty.cc:
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

class FieldSetNrrdProperty : public Module {
public:
  FieldSetNrrdProperty(GuiContext*);

  virtual ~FieldSetNrrdProperty();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guinrrd1name_;
  GuiString     guinrrd2name_;
  GuiString     guinrrd3name_;
  GuiInt        guinrrd1usename_;
  GuiInt        guinrrd2usename_;
  GuiInt        guinrrd3usename_;
};


DECLARE_MAKER(FieldSetNrrdProperty)
  FieldSetNrrdProperty::FieldSetNrrdProperty(GuiContext* ctx)
    : Module("FieldSetNrrdProperty", ctx, Source, "Fields", "CardioWave"),
      guinrrd1name_(ctx->subVar("nrrd1-name")),
      guinrrd2name_(ctx->subVar("nrrd2-name")),
      guinrrd3name_(ctx->subVar("nrrd3-name")),
      guinrrd1usename_(ctx->subVar("nrrd1-usename")),
      guinrrd2usename_(ctx->subVar("nrrd2-usename")),
      guinrrd3usename_(ctx->subVar("nrrd3-usename"))
{
}

FieldSetNrrdProperty::~FieldSetNrrdProperty(){
}

void
FieldSetNrrdProperty::execute()
{
  string nrrd1name = guinrrd1name_.get();
  string nrrd2name = guinrrd2name_.get();
  string nrrd3name = guinrrd3name_.get();
  int nrrd1usename = guinrrd1usename_.get();
  int nrrd2usename = guinrrd2usename_.get();
  int nrrd3usename = guinrrd3usename_.get();
    
  FieldHandle handle;
  FieldIPort  *iport;
  FieldOPort *oport;
  NrrdDataHandle fhandle;
  NrrdIPort *ifport;
        
  if(!(iport = static_cast<FieldIPort *>(getIPort("field"))))
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
  
  // Scan nrrd input port 1
  if (!(ifport = static_cast<NrrdIPort *>(getIPort("nrrd1"))))
    {
      error("Could not find nrrd 1 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd1usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              nrrd1name = name;
              guinrrd1name_.set(name);
              ctx->reset();
            }
        }
      handle->set_property(nrrd1name,fhandle,false);
    }

  // Scan nrrd input port 2     
  if (!(ifport = static_cast<NrrdIPort *>(getIPort("nrrd2"))))
    {
      error("Could not find nrrd 2 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd2usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {    
              nrrd2name = name;
              guinrrd2name_.set(name);
              ctx->reset();
            }    
        }

      handle->set_property(nrrd2name,fhandle,false);
    }

  // Scan nrrd input port 3     
  if (!(ifport = static_cast<NrrdIPort *>(getIPort("nrrd3"))))
    {
      error("Could not find nrrd 3 input port");
      return;
    }
        
  if (ifport->get(fhandle))
    {
      if (nrrd3usename)
        {
          string name;
          fhandle->get_property("name",name);
          if (name != "") 
            {
              nrrd3name = name;
              guinrrd1name_.set(name);
              ctx->reset();
            }
        }
    
      handle->set_property(nrrd3name,fhandle,false);
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
FieldSetNrrdProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




