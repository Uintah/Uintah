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
 *  CreateParametersBundle.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Bundle/Bundle.h>
#include <vector>
#include <string>


// We may move this to a more general spot in the SCIRun tree
namespace SCIRun {

using namespace SCIRun;

class CreateParametersBundle : public Module {
public:
  CreateParametersBundle(GuiContext*);

  virtual ~CreateParametersBundle();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
	std::vector<std::string> converttcllist(std::string str);

  GuiString  guiparnames_;
  GuiString  guiparvalues_;
  GuiString  guipartypes_;
  GuiString  guipardescriptions_;
  GuiString  guisynchronise_;
  
  std::vector<std::string> parnames_;
  std::vector<std::string> parvalues_;
  std::vector<std::string> partypes_;
  

};


DECLARE_MAKER(CreateParametersBundle)
CreateParametersBundle::CreateParametersBundle(GuiContext* ctx)
  : Module("CreateParametersBundle", ctx, Source, "Tools", "CardioWave"),
  guiparnames_(get_ctx()->subVar("par-names")),
  guiparvalues_(get_ctx()->subVar("par-values")),
  guipartypes_(get_ctx()->subVar("par-types")),
  guipardescriptions_(get_ctx()->subVar("par-descriptions")),
  guisynchronise_(get_ctx()->subVar("synchronise"))  
{
}

CreateParametersBundle::~CreateParametersBundle(){
}

void CreateParametersBundle::execute()
{
  BundleOPort *oport;
  
  get_gui()->lock();
  get_gui()->execute(guisynchronise_.get());
  get_gui()->unlock();
  
  if (!(oport = static_cast<BundleOPort *>(get_oport("Parameters"))))
  {
    error("Could not find output port");
  }

  parnames_ = converttcllist(guiparnames_.get());
  partypes_ = converttcllist(guipartypes_.get());
  parvalues_ = converttcllist(guiparvalues_.get());

  BundleHandle handle;
  handle = scinew Bundle;
  
  
  if ((parnames_.size() != parvalues_.size())&&(partypes_.size() != parvalues_.size()))
  {
    error("internal error in module, TCL lists are not of same length");
    return;
  }
  
  for (size_t p=0; p < parnames_.size(); p++)
  {

    if (partypes_[p] == "string")
    {
      StringHandle str = scinew String(parvalues_[p]);
      handle->setString(parnames_[p],str);  
    }
    else if (partypes_[p] == "double")
    {
      MatrixHandle scalar = dynamic_cast<Matrix*>(scinew DenseMatrix(1,1));
      std::istringstream iss(parvalues_[p]);
      double val; iss >> val;
      scalar->put(0,0,val);
      handle->setMatrix(parnames_[p],scalar);
    }
    else if (partypes_[p] == "float")
    {
      MatrixHandle scalar = dynamic_cast<Matrix*>(scinew DenseMatrix(1,1));
      std::istringstream iss(parvalues_[p]);
      double val; iss >> val;
      scalar->put(0,0,val);
      handle->setMatrix(parnames_[p],scalar);
    }
    else if (partypes_[p] == "integer")
    {
      MatrixHandle scalar = dynamic_cast<Matrix*>(scinew DenseMatrix(1,1));
      std::istringstream iss(parvalues_[p]);
      double val; iss >> val;
      scalar->put(0,0,val);
      handle->setMatrix(parnames_[p],scalar);
    }
  }
 
  oport->send(handle);
 
}

void
 CreateParametersBundle::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


std::vector<std::string> CreateParametersBundle::converttcllist(std::string str)
{
	std::string result;
	std::vector<std::string> list(0);
	long lengthlist = 0;
	
	// Yeah, it is TCL dependent:
	// TCL::llength determines the length of the list
	get_gui()->lock();
	get_gui()->eval("llength { "+str + " }",result);	
	istringstream iss(result);
	iss >> lengthlist;
	get_gui()->unlock();
	if (lengthlist < 0) return(list);
	
	list.resize(lengthlist);
	get_gui()->lock();
	for (long p = 0;p<lengthlist;p++)
	{
		ostringstream oss;
		// TCL dependency:
		// TCL::lindex retrieves the p th element from the list
		oss << "lindex { " << str <<  " } " << p;
		get_gui()->eval(oss.str(),result);
		list[p] = result;
	}
	get_gui()->unlock();
	return(list);
}



} // End namespace SCIRun


