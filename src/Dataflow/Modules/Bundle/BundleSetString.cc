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
 *  BundleSetString.cc:
 *
 *  Written by:
 *   jeroen
 *
 */

#include <Core/Bundle/Bundle.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace std;

class BundleSetString : public Module {
public:
  BundleSetString(GuiContext*);

  virtual ~BundleSetString();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     guistring1name_;
  GuiString     guistring2name_;
  GuiString     guistring3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetString)
  BundleSetString::BundleSetString(GuiContext* ctx)
    : Module("BundleSetString", ctx, Source, "Bundle", "SCIRun"),
      guistring1name_(ctx->subVar("string1-name")),
      guistring2name_(ctx->subVar("string2-name")),
      guistring3name_(ctx->subVar("string3-name")),
      guibundlename_(ctx->subVar("bundlename"))
{
}

BundleSetString::~BundleSetString(){
}

void
BundleSetString::execute()
{
  string string1name = guistring1name_.get();
  string string2name = guistring2name_.get();
  string string3name = guistring3name_.get();
  string bundlename = guibundlename_.get();
    
  BundleHandle handle, oldhandle;
  BundleIPort  *iport;
  BundleOPort *oport;
  StringHandle fhandle;
  StringIPort *ifport;
        
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
  if (!(ifport = static_cast<StringIPort *>(get_iport("string1"))))
  {
    error("Could not find string 1 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
      handle->setString(string1name,fhandle);
  }

  // Scan string input port 2       
  if (!(ifport = static_cast<StringIPort *>(get_iport("string2"))))
  {
    error("Could not find string 2 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->setString(string2name,fhandle);
  }

  // Scan string input port 3       
  if (!(ifport = static_cast<StringIPort *>(get_iport("string3"))))
  {
    error("Could not find string 3 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  { 
    handle->setString(string3name,fhandle);
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
BundleSetString::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

