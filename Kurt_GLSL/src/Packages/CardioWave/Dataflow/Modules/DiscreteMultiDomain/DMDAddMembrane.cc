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
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/StringPort.h>
#include <Packages/CardioWave/Core/XML/SynapseXML.h>
#include <Packages/ModelCreation/Core/Converter/ConverterAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWave {

using namespace SCIRun;

class DMDAddMembrane : public Module {
public:
  DMDAddMembrane(GuiContext*);

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

  SynapseXML synapsexml_;

};


DECLARE_MAKER(DMDAddMembrane)

DMDAddMembrane::DMDAddMembrane(GuiContext* ctx)
  : Module("DMDAddMembrane", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guimembranenames_(ctx->subVar("mem-names")),
    guimembranename_(ctx->subVar("mem-name")),
    guimembraneparam_(ctx->subVar("mem-param")),
    guimembranedesc_(ctx->subVar("mem-desc"))
{
}

void DMDAddMembrane::execute()
{
  ModelCreation::ConverterAlgo mc(this);

  BundleIPort* membranebundle_iport = dynamic_cast<BundleIPort*>(getIPort(0));
  if (membranebundle_iport == 0)
  {
    error("Could not find membrane input port");
    return;
  }
  
  int membrane_nodetype = 0;
  
  BundleHandle MembraneBundle;
  BundleHandle Membrane;
  MatrixHandle NodeType;
  
  if (membranebundle_iport->get(MembraneBundle))
  {
    // In case we already have a few other membranes lined up
    
    // Determine the nodetype numbers already used.
    int numbundles = MembraneBundle->numBundles();
    for (size_t p = 0; p< numbundles; p++)
    {
      Membrane = MembraneBundle->getBundle(MembraneBundle->getBundleName(p));
      if (Membrane.get_rep())
      {
        NodeType = Membrane->getMatrix("nodetype");
        if (NodeType.get_rep())
        {
          int nodetype;
          if(mc.MatrixToInt(NodeType,nodetype))
          {
            if (nodetype >= membrane_nodetype) membrane_nodetype = nodetype + 1;
          }
        }
      }
    }
  }
  else
  {
    // Create a new output bundle
  
    MembraneBundle = scinew Bundle();
    if (MembraneBundle.get_rep() == 0)
    {
      error("Could not allocate new membrane bundle");
      return;
    } 
  }

  // Add a new bundle to the bundle with the data
  // from this module
  Membrane = scinew Bundle();
  if (Membrane.get_rep() == 0)
  {
    error("Could not allocate new membrane bundle");
    return;
  }
  
  std::ostringstream oss;
  oss << "membrane_" << membrane_nodetype; 
  MembraneBundle->setBundle(oss.str(),Membrane);
  
  if(!(mc.IntToMatrix(membrane_nodetype,NodeType)))
  {
    error("Could not build nodetype matrix");
    return;  
  }
  
  Membrane->setMatrix("nodetype",NodeType);
  
  FieldIPort* geometryport = dynamic_cast<FieldIPort*>(getIPort(1));
  if (geometryport == 0)
  {
    error("Could not find Membrane Geometry port");
    return;
  }

  FieldHandle Geometry;
  geometryport->get(Geometry);
  
  if (Geometry.get_rep() == 0)
  {
    error("Membrane Geometry field is empty");
    return;  
  }

  Membrane->setField("geometry",Geometry);
 
  StringIPort* paramport = dynamic_cast<StringIPort*>(getIPort(2));
  if (paramport == 0)
  {
    error("Could not find Paramerter input port");
    return;
  } 

  std::string cmd;
  cmd = getID() + " get_param";
  gui->lock();
  gui->eval(cmd);
  gui->unlock();
  ctx->reset();
  
  std::string paramstr;
 
  paramstr = guimembraneparam_.get();
  StringHandle Param_from_port;
  paramport->get(Param_from_port);
  if (Param_from_port.get_rep()) 
  {
    paramstr += Param_from_port->get();
  }

  StringHandle Param = scinew String(guimembraneparam_.get());
  if (Param.get_rep() == 0)
  {
    error("Could not create parameter string");
    return;
  } 
  
  Membrane->setString("parameters",Param);
  
  std::string membranename = guimembranename_.get();
  StringHandle MembraneName = scinew String(membranename);
  Membrane->setString("name",MembraneName);
  
  SynapseItem item = synapsexml_.get_synapse(membranename);
  StringHandle NodeTypeName = scinew String(item.nodetype);
  Membrane->setString("nodetype_name",NodeTypeName);
  
  if (membranename != "")
  {
    BundleOPort* oport = dynamic_cast<BundleOPort*>(getOPort(0));
    if (oport)
    {
      oport->send(MembraneBundle);
    }  
  }
}


void DMDAddMembrane::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() > 1)
  {
    if (args[1] == "get_membrane_names")
    {
      std::vector<std::string> names = synapsexml_.get_names(); 
      guimembranenames_.set(convertclist(names));
    }
    else if (args[1] == "set_membrane")
    {
      if (args[2] != "")
      {
        SynapseItem item = synapsexml_.get_synapse(args[2]);
        std::string param = item.parameters;
        std::string desc  = item.description;
        guimembranename_.set(args[2]);
        guimembraneparam_.set(param);
        guimembranedesc_.set(desc);
        ctx->reset();
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


std::string DMDAddMembrane::totclstring(std::string &instring)
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

std::string DMDAddMembrane::convertclist(std::vector<std::string> list)
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


