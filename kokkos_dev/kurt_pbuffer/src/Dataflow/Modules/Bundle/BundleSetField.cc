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
 *  BundleSetField.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetField : public Module {
public:
  BundleSetField(GuiContext*);

  virtual ~BundleSetField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guifield1name_;
  GuiString     guifield2name_;
  GuiString     guifield3name_;
  GuiInt        guifield1usename_;
  GuiInt        guifield2usename_;
  GuiInt        guifield3usename_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetField)
  BundleSetField::BundleSetField(GuiContext* ctx)
    : Module("BundleSetField", ctx, Source, "Bundle", "SCIRun"),
      guifield1name_(ctx->subVar("field1-name")),
      guifield2name_(ctx->subVar("field2-name")),
      guifield3name_(ctx->subVar("field3-name")),
      guifield1usename_(ctx->subVar("field1-usename")),
      guifield2usename_(ctx->subVar("field2-usename")),
      guifield3usename_(ctx->subVar("field3-usename")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetField::~BundleSetField(){
}

void
BundleSetField::execute()
{
  string field1name = guifield1name_.get();
  string field2name = guifield2name_.get();
  string field3name = guifield3name_.get();
  int field1usename = guifield1usename_.get();
  int field2usename = guifield2usename_.get();
  int field3usename = guifield3usename_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  FieldHandle fhandle;
  FieldIPort *ifport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("bundle"))))
    {
      error("Could not find bundle input port");
      return;
    }
        
  // Create a new bundle
  // Since a bundle consists of only handles we can copy
  // it several times without too much memory overhead
  if (iport->get(oldhandle))
    {   // Copy all the handles from the existing bundle
      handle = oldhandle->clone();
    }
  else
    {   // Create a brand new bundle
      handle = scinew Bundle;
    }
        
  // Scan bundle input port 1
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
      handle->setField(field1name,fhandle);
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

      handle->setField(field2name,fhandle);
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
    
      handle->setField(field3name,fhandle);
    }
        
  // Now post the output
        
  if (!(oport = static_cast<BundleOPort *>(get_oport("bundle"))))
    {
      error("Could not find bundle output port");
      return;
    }
    
  if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }
        
  oport->send(handle);
}

void
BundleSetField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


