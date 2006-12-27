
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
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class GetStringsFromBundle : public Module {
public:
  GetStringsFromBundle(GuiContext*);
  virtual void execute();
  
private:
  GuiString             guistring1name_;
  GuiString             guistring2name_;
  GuiString             guistring3name_;
  GuiString             guistrings_;
};


DECLARE_MAKER(GetStringsFromBundle)

GetStringsFromBundle::GetStringsFromBundle(GuiContext* ctx)
  : Module("GetStringsFromBundle", ctx, Filter, "Bundle", "SCIRun"),
    guistring1name_(get_ctx()->subVar("string1-name"), "string1"),
    guistring2name_(get_ctx()->subVar("string2-name"), "string2"),
    guistring3name_(get_ctx()->subVar("string3-name"), "string3"),
    guistrings_(get_ctx()->subVar("string-selection"), "")
{
}

void
GetStringsFromBundle::execute()
{
  // Define input handle:
  BundleHandle handle;
  
  // Get data from input port:
  if (!(get_input_handle("bundle",handle,true))) return;
  
  if (inputs_changed_ || guistring1name_.changed() || guistring2name_.changed() ||
      guistring3name_.changed() || !oport_cached("bundle") || !oport_cached("string1") ||
       !oport_cached("string2") || !oport_cached("string3"))
  {
    StringHandle fhandle;
    std::string string1name = guistring1name_.get();
    std::string string2name = guistring2name_.get();
    std::string string3name = guistring3name_.get();
    std::string stringlist;
        
    int numStrings = handle->numStrings();
    for (int p = 0; p < numStrings; p++)
    {
      stringlist += "{" + handle->getStringName(p) + "} ";
    }

    guistrings_.set(stringlist);
    get_ctx()->reset();
 
    // Send string1 if we found one that matches the name:
    if (handle->isString(string1name))
    {
      fhandle = handle->getString(string1name);
      send_output_handle("string1",fhandle,false);
    } 

    // Send string2 if we found one that matches the name:
    if (handle->isString(string2name))
    {
      fhandle = handle->getString(string2name);
      send_output_handle("string2",fhandle,false);
    } 

    // Send string3 if we found one that matches the name:
    if (handle->isString(string3name))
    {
      fhandle = handle->getString(string3name);
      send_output_handle("string3",fhandle,false);
    } 
    
    send_output_handle("bundle",handle,false);    
  }      
}



