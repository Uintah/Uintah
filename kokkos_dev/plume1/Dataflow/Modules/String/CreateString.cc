//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  

#include <Core/Datatypes/String.h>
#include <Dataflow/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

using namespace SCIRun;

class CreateString : public Module {
public:
  CreateString(GuiContext*);

  virtual ~CreateString();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:

  GuiString  inputstring_;
  GuiString  getinputstring_;
};


DECLARE_MAKER(CreateString)
CreateString::CreateString(GuiContext* ctx)
  : Module("CreateString", ctx, Source, "String", "SCIRun"),
  inputstring_(ctx->subVar("inputstring")),
  getinputstring_(ctx->subVar("get-inputstring"))
{
}

CreateString::~CreateString()
{
}

void CreateString::execute()
{
  gui->lock();
  gui->execute(getinputstring_.get());
  gui->unlock();

  StringOPort* oport;
  if (!(oport = dynamic_cast<StringOPort *>(get_oport(0))))
  {
    error("could not find output port");
    return;
  }
  
  // TCL HAS A TENDENCY TO ADD A LINEFEED AT THE END
  std::string str = inputstring_.get();
  if((str.size() > 0)&&(str[str.size()-1] == '\n')) str = str.substr(0,str.size()-1); 
  StringHandle handle = reinterpret_cast<String *>(scinew String(str));
  oport->send(handle);
}

void
 CreateString::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


