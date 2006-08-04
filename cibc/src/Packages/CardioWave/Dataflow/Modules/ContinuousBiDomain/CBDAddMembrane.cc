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


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/String.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWave/Core/XML/MembraneXML.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWave {

using namespace SCIRun;

class CBDAddMembrane : public Module {
public:
  CBDAddMembrane(GuiContext*);

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:  
  // TCL tools
  std::string totclstring(std::string &instring);
  std::string convertclist(std::vector<std::string> list);

  GuiString guimembranenames_;
  GuiString guimembranename_;
  GuiString guimembraneparam_;
  GuiString guimembranedesc_;
  GuiInt    guimembranetype_;

  MembraneXML membranexml_;

};


DECLARE_MAKER(CBDAddMembrane)

CBDAddMembrane::CBDAddMembrane(GuiContext* ctx)
  : Module("CBDAddMembrane", ctx, Source, "ContinuousBiDomain", "CardioWave"),
    guimembranenames_(get_ctx()->subVar("mem-names")),
    guimembranename_(get_ctx()->subVar("mem-name")),
    guimembraneparam_(get_ctx()->subVar("mem-param")),
    guimembranedesc_(get_ctx()->subVar("mem-desc")),
    guimembranetype_(get_ctx()->subVar("mem-type"))
{
    std::string defaultname = membranexml_.get_default_name();
    guimembranename_.set(defaultname);
}

void CBDAddMembrane::execute()
{
  BundleHandle MembraneBundle;
  BundleHandle Membrane;
  MatrixHandle MembraneType;
  StringHandle Parameters_from_port;
  
  get_input_handle("MembraneBundle",MembraneBundle,false);
  get_input_handle("MembraneType",MembraneType,false);
  get_input_handle("Parameters",Parameters_from_port,false);
  
  if (MembraneBundle.get_rep() == 0)
  {
    MembraneBundle = scinew Bundle();
    if (MembraneBundle.get_rep() == 0)
    {
      error("Could not allocate new membrane bundle");
      return;
    }   
  }
  else
  {
    MembraneBundle.detach();    
  }

  int membrane_num = 0;
  std::string fieldname;
  {
    std::ostringstream oss;
    oss << "Membrane_" << membrane_num;
    fieldname = oss.str(); 
  }
  while (MembraneBundle->isBundle(fieldname))
  {
    membrane_num++;
    {
      std::ostringstream oss;
      oss << "Membrane_" << membrane_num;
      fieldname = oss.str(); 
    }
  }

  
  SCIRunAlgo::ConverterAlgo mc(this);

  if (MembraneType.get_rep())
  {
    int memtype;
    if(mc.MatrixToInt(MembraneType,memtype)) guimembranetype_.set(memtype);
    get_ctx()->reset();
  }

  
  // Add a new bundle to the bundle with the data
  // from this module
  Membrane = scinew Bundle();
  if (Membrane.get_rep() == 0)
  {
    error("Could not allocate new membrane bundle");
    return;
  }
  
  {
    std::ostringstream oss;
    oss << "Membrane_" << membrane_num;
    fieldname = oss.str(); 
  }
  MembraneBundle->setBundle(fieldname,Membrane);
 
 
  std::string cmd;
  cmd = get_id() + " get_param";
  get_gui()->lock();
  get_gui()->eval(cmd);
  get_gui()->unlock();
  get_ctx()->reset();
  
  std::string paramstr; 
  paramstr = guimembraneparam_.get();
  if (Parameters_from_port.get_rep()) 
  {
    paramstr += "\n";
    paramstr += Parameters_from_port->get();
    paramstr += "\n";
  }

  std::string membranename = guimembranename_.get();
  StringHandle MembraneName = scinew String(membranename);
  Membrane->setString("Name",MembraneName);
  
  int membranetype = guimembranetype_.get();
  
  MembraneItem item = membranexml_.get_membrane(membranename);
  
  {
    std::ostringstream oss;
    oss << "\n" << item.nodetype << " = " << membranetype << "\n";
    paramstr += oss.str();
  }
  
  StringHandle Parameters = scinew String(paramstr);
  if (Parameters.get_rep() == 0)
  {
    error("Could not create parameter string");
    return;
  } 
  Membrane->setString("Parameters",Parameters);
    
  StringHandle SourceFile = scinew String(item.sourcefile);
  Membrane->setString("SourceFile",SourceFile);
  
  send_output_handle("MembraneBundle",MembraneBundle,true);
}


void CBDAddMembrane::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() > 1)
  {
    if (args[1] == "get_membrane_names")
    {
      std::vector<std::string> names = membranexml_.get_names(); 
      guimembranenames_.set(convertclist(names));
    }
    else if (args[1] == "set_membrane")
    {
      if (args[2] != "")
      {
        MembraneItem item = membranexml_.get_membrane(args[2]);
        std::string param = item.parameters;
        std::string desc  = item.description;
        guimembranename_.set(args[2]);
        guimembraneparam_.set(param);
        guimembranedesc_.set(desc);
        get_ctx()->reset();
      }
    }
    else
    {
      Module::tcl_command(args, userdata);
    }
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}


std::string CBDAddMembrane::totclstring(std::string &instring)
{
	int strsize = instring.size();
	int specchar = 0;
	for (int p = 0; p < strsize; p++)
		if ((instring[p]=='\n')||(instring[p]=='\t')||(instring[p]=='\b')||(instring[p]=='\r')||(instring[p]=='{')||(instring[p]=='}')
				||(instring[p]=='[')||(instring[p]==']')||(instring[p]=='\\')||(instring[p]=='$')||(instring[p]=='"')) specchar++;
	
	std::string newstring;
	newstring.resize(strsize+specchar);
	int q = 0;

	for (int p = 0; p < strsize; p++)
	{
		if (instring[p]=='\n') { newstring[q++] = '\\'; newstring[q++] = 'n'; continue; }
		if (instring[p]=='\t') { newstring[q++] = '\\'; newstring[q++] = 't'; continue; }
		if (instring[p]=='\b') { newstring[q++] = '\\'; newstring[q++] = 'b'; continue; }
		if (instring[p]=='\r') { newstring[q++] = '\\'; newstring[q++] = 'r'; continue; }
		if (instring[p]=='{')  { newstring[q++] = '\\'; newstring[q++] = '{'; continue; }
		if (instring[p]=='}')  { newstring[q++] = '\\'; newstring[q++] = '}'; continue; }
		if (instring[p]=='[')  { newstring[q++] = '\\'; newstring[q++] = '['; continue; }
		if (instring[p]==']')  { newstring[q++] = '\\'; newstring[q++] = ']'; continue; }
		if (instring[p]=='\\') { newstring[q++] = '\\'; newstring[q++] = '\\'; continue; }
		if (instring[p]=='$')  { newstring[q++] = '\\'; newstring[q++] = '$'; continue; }
		if (instring[p]=='"')  { newstring[q++] = '\\'; newstring[q++] = '"'; continue; }
		newstring[q++] = instring[p];
	}
	
  newstring = "{" + newstring + "}";
	return(newstring);
}

std::string CBDAddMembrane::convertclist(std::vector<std::string> list)
{
  std::string result;
  for (size_t p=0; p < list.size(); p++)
  {
      result += totclstring(list[p]);
      result += " ";
  }
  return(result);
}

} // End namespace CardioWave


