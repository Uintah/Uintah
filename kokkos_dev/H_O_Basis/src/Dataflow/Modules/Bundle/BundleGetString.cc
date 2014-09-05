
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
 *  BundleGetString.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/StringPort.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleGetString : public Module {
public:
  BundleGetString(GuiContext*);

  virtual ~BundleGetString();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guistring1name_;
  GuiString             guistring2name_;
  GuiString             guistring3name_;
  GuiString             guistrings_;
};


DECLARE_MAKER(BundleGetString)
  BundleGetString::BundleGetString(GuiContext* ctx)
    : Module("BundleGetString", ctx, Source, "Bundle", "SCIRun"),
      guistring1name_(ctx->subVar("string1-name")),
      guistring2name_(ctx->subVar("string2-name")),
      guistring3name_(ctx->subVar("string3-name")),
      guistrings_(ctx->subVar("string-selection"))
{

}

BundleGetString::~BundleGetString(){
}

void BundleGetString::execute()
{
  string string1name = guistring1name_.get();
  string string2name = guistring2name_.get();
  string string3name = guistring3name_.get();
  string stringlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort *oport;
  StringHandle fhandle;
  StringOPort *ofport;
        
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

  int numStrings = handle->numStrings();
  for (int p = 0; p < numStrings; p++)
  {
    stringlist += "{" + handle->getStringName(p) + "} ";
  }


  if (handle.get_rep() == 0)
  {   
    warning("Empty bundle connected to the input port");
    return;
  }


  guistrings_.set(stringlist);
  ctx->reset();

 
  if (!(ofport = static_cast<StringOPort *>(get_oport("string1"))))
  {
    error("Could not find string 1 output port");
    return; 
  }
  
  if (handle->isString(string1name))
  {
    fhandle = handle->getString(string1name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<StringOPort *>(get_oport("string2"))))
  {
    error("Could not find string 2 output port");
    return; 
  }
 
  if (handle->isString(string2name))
  {
    fhandle = handle->getString(string2name);
    ofport->send(fhandle);
  }
      
 
  if (!(ofport = static_cast<StringOPort *>(get_oport("string3"))))
  {
    error("Could not find string 3 output port");
    return; 
  }

  if (handle->isString(string3name))
  {
    fhandle = handle->getString(string3name);
    ofport->send(fhandle);
  }
        
  if ((oport = static_cast<BundleOPort *>(get_oport("bundle"))))
  {
    oport->send(handle);
  }
        
}

void BundleGetString::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


