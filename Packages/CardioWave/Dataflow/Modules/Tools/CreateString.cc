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
 *  CreateString.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h> 

namespace SCIRun {

using namespace SCIRun;

class CreateString : public Module {
public:
  CreateString(GuiContext*);

  virtual ~CreateString();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString     inputstring_;
  GuiString     labelstring_;
};


DECLARE_MAKER(CreateString)
CreateString::CreateString(GuiContext* ctx)
  : Module("CreateString", ctx, Source, "Tools", "CardioWave"),
  inputstring_(ctx->subVar("inputstring")),
  labelstring_(ctx->subVar("labelstring"))
{
}

CreateString::~CreateString(){
}

void CreateString::execute()
{
    std::string str = inputstring_.get();
    NrrdOPort *oport;
    
	if(!(oport = static_cast<NrrdOPort *>(get_oport("string"))))
	{
		error("Could not find nrrdstring output port");
		return;
	}    
    
    NrrdDataHandle nrrdH = scinew NrrdData();
    if (nrrdH == 0)
    {
        error("Could not allocate NrrdData");
        return;
    }
    
    if (str.size() > 0)
    {
        // Need to add some error checking here
        nrrdAlloc(nrrdH->nrrd, nrrdTypeChar, 1, static_cast<int>(str.size()));
        nrrdAxisInfoSet(nrrdH->nrrd, nrrdAxisInfoLabel, "nrrdstring");
        nrrdH->nrrd->axis[0].kind = nrrdKindDomain;
        char *val = (char*)nrrdH->nrrd->data;
        if (val == 0)
        {
            error("Could not allocate space for nrrdstring");
            return;
        }
        for (size_t p=0;p<str.size();p++) val[p] =str[p];
    }
    
    std::string label = labelstring_.get();
    if (label != "") nrrdH->set_property("name",label,false);
    
    oport->send(nrrdH);
}

void
 CreateString::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


