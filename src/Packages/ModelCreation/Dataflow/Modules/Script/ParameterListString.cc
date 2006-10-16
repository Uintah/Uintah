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

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;

class ParameterListString : public Module {
public:
  ParameterListString(GuiContext*);
  virtual void execute();
    
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


void
ParameterListString::execute()
{
  BundleHandle bundle;
  StringHandle handle;
  
  get_input_handle("ParameterList",bundle,false);
  get_input_handle("String",handle,false);

  if (inputs_changed_ || guistringname_.changed()  || !oport_cached("ParameterList") || !oport_cached("String"))
  {
    std::string stringname = guistringname_.get();
    std::string handlelist;
        
    if (bundle.get_rep() == 0)
    {   
      bundle = scinew Bundle();
    }

    if (handle.get_rep() != 0)
    {
      bundle = bundle->clone();
      bundle->setString(stringname,handle);
      handle = 0;
    }

    // Update the GUI with all the handle names
    // So the can select which one he or she wants
    // to extract
    
    size_t numstrings = bundle->numStrings();
    for (size_t p = 0; p < numstrings; p++)
    {
      handlelist += "{" + bundle->getStringName(p) + "} ";
    }
    guistrings_.set(handlelist);
    get_ctx()->reset();

    if (bundle->isString(stringname))
    {
      handle = bundle->getString(stringname);
      send_output_handle("String",handle,false);
    }        

    send_output_handle("ParameterList",bundle,false);
  }
}

} // End namespace ModelCreation


