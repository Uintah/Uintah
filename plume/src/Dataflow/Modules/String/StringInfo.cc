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

class StringInfo : public Module {
public:
  StringInfo(GuiContext*);

  virtual ~StringInfo();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString  inputstring_;
  GuiString  update_;
  
};



DECLARE_MAKER(StringInfo)
StringInfo::StringInfo(GuiContext* ctx)
  : Module("StringInfo", ctx, Source, "String", "SCIRun"),
    inputstring_(ctx->subVar("inputstring")),
    update_(ctx->subVar("update"))    
{
}

StringInfo::~StringInfo()
{
}

void StringInfo::execute()
{

 StringIPort* iport;
 if (!(iport = dynamic_cast<StringIPort *>(get_iport(0))))
  {
    error("could not find input port");
    return;
  }
  
  StringHandle handle;
  iport->get(handle);
  
  if (handle.get_rep() == 0)
  {
    inputstring_.set("<empty string>");
    return;
  }
  else
  {
    inputstring_.set(handle->get());
  }

  gui->lock();
  gui->execute(update_.get());
  gui->unlock();
}

void
 StringInfo::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


