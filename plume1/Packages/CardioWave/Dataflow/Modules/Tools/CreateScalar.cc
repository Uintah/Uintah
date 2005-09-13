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
 *  CreateScalar.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h> 
#include <Core/Datatypes/NrrdScalar.h>

namespace SCIRun {

using namespace SCIRun;

class CreateScalar : public Module {
public:
  CreateScalar(GuiContext*);

  virtual ~CreateScalar();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     inputscalar_;
  GuiString     labelstring_;
  GuiString     scalartype_;
};


DECLARE_MAKER(CreateScalar)
CreateScalar::CreateScalar(GuiContext* ctx)
  : Module("CreateScalar", ctx, Source, "Tools", "CardioWave"),
  inputscalar_(ctx->subVar("inputscalar")),
  labelstring_(ctx->subVar("labelstring")),
  scalartype_(ctx->subVar("scalartype"))
{
}

CreateScalar::~CreateScalar(){
}

void CreateScalar::execute()
{
    std::string str = inputscalar_.get();
    std::string type = scalartype_.get();
    NrrdOPort *oport;
    
	if(!(oport = static_cast<NrrdOPort *>(get_oport("nrrdvalue"))))
	{
		error("Could not find nrrdvalue output port");
		return;
	}    
    
    if (str.size() == 0)
    {
        error("No value has been entered");
        return;
    }
    
    NrrdScalar scalar(str,type);
    oport->send(scalar.gethandle());
}

void
 CreateScalar::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


