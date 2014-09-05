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
 * FILE: ParameterListString.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 17 SEP 2005
 */ 

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/StringPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class ParameterListString : public Module {
public:
  ParameterListString(GuiContext*);

  virtual ~ParameterListString();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:

  GuiString             guistringname_;
  GuiString             guistrings_;  
};


DECLARE_MAKER(ParameterListString)
ParameterListString::ParameterListString(GuiContext* ctx)
  : Module("ParameterListString", ctx, Source, "Script", "ModelCreation"),
    guistringname_(get_ctx()->subVar("string-name")),
    guistrings_(get_ctx()->subVar("string-selection"))
{
}

ParameterListString::~ParameterListString()
{
}

void ParameterListString::execute()
{
  string stringname = guistringname_.get();
  string stringlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort  *oport;
  
  StringHandle fhandle;
  StringIPort *ifport;
  StringOPort *ofport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("ParameterList"))))
  {
    error("Cannot not find ParameterList input port");
    return;
  }

  if(!(ifport = static_cast<StringIPort *>(get_iport("String"))))
  {
    error("Cannot not find String input port");
    return;
  }

  // If no input bundle is found, create a new one

  iport->get(handle);
  if (handle.get_rep() == 0)
  {   
    handle = dynamic_cast<Bundle *>(scinew Bundle());
  }

  ifport->get(fhandle);
  if (fhandle.get_rep() != 0)
  {
    handle = handle->clone();
    handle->setString(stringname,fhandle);
    fhandle = 0;
  }

  // Update the GUI with all the string names
  // So the can select which one he or she wants
  // to extract
  
  size_t numstrings = handle->numStrings();
  for (size_t p = 0; p < numstrings; p++)
  {
    stringlist += "{" + handle->getStringName(p) + "} ";
  }
  guistrings_.set(stringlist);
  get_ctx()->reset();


  if (!(ofport = static_cast<StringOPort *>(get_oport("String"))))
  {
    error("Could not find String output port");
    return; 
  }
  if (!(oport = static_cast<BundleOPort *>(get_oport("ParameterList"))))
  {
    error("Could not find ParameterList output port");
    return; 
  }
 
  if (handle->isString(stringname))
  {
    fhandle = handle->getString(stringname);
    ofport->send(fhandle);
  }        
          
  oport->send(handle);

}


void
 ParameterListString::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


