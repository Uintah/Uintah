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


#include <Dataflow/Network/Module.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/String.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWave/Core/XML/SynapseXML.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>

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
    guimembranenames_(get_ctx()->subVar("mem-names")),
    guimembranename_(get_ctx()->subVar("mem-name")),
    guimembraneparam_(get_ctx()->subVar("mem-param")),
    guimembranedesc_(get_ctx()->subVar("mem-desc"))
{
    std::string defaultname = synapsexml_.get_default_name();
    guimembranename_.set(defaultname);
}

void DMDAddMembrane::execute()
{
  // Define input handles:
  BundleHandle MembraneBundle;
  BundleHandle Membrane;
  FieldHandle  Geometry;
  StringHandle Parameters_from_port;
  
  // Obtain the input from the ports:
  if (!(get_input_handle("Geometry",Geometry,true))) return;  
  get_input_handle("MembraneBundle",MembraneBundle,false);
  get_input_handle("Parameters",Parameters_from_port,false);

  // Make sure we have the latest from TCL
  std::string cmd;
  cmd = get_id() + " get_param";
  get_gui()->lock();
  get_gui()->eval(cmd);
  get_gui()->unlock();
  get_ctx()->reset();
  
  // Only execute if something has changed:
  if (inputs_changed_  || guimembranename_.changed() || 
      guimembraneparam_.changed() || !oport_cached("MembraneBundle"))
  {
    // Create access point to field algorithms
    SCIRunAlgo::FieldsAlgo algo(this);

    int numnodes, numelems;
    if (!(algo.GetFieldInfo(Geometry,numnodes,numelems)))
    {
      error("DMDAddMembrane: Could not determine number of elements/nodes in the field");
      return;        
    }
    
    if ((numnodes == 0) || (numelems == 0))
    {
      send_output_handle("MembraneBundle",MembraneBundle,false);
      return;    
    } 
    
    // Empty the geometry and make it linear, we need a linear version
    // to project data on:
    if (!(algo.ClearAndChangeFieldBasis(Geometry,Geometry,"Linear")))
    {
      error("DMDAddMembrane: Could not build a linear field for the membrane");
      return;        
    }
    
    // If we have an input bundle use that one otherwise create a new one:
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
      // We use the old one, but need to create a new copy for the data flow:
      // Luckily we are using bundles and only the bundle wrapper changes and
      // not the underlying objects:
      MembraneBundle.detach();    
    }

    std::string membranename = guimembranename_.get();
    StringHandle MembraneName = scinew String(membranename);

    // Find what the next index will be for the membranes
    // We start with 1, so 0 means no membrane:
    int membrane_num = 1;
    std::string fieldname;
    {
      std::ostringstream oss;
      oss << "Membrane_" << membrane_num;
      fieldname = oss.str(); 
    }
    while (MembraneBundle->isBundle(fieldname))
    {
      BundleHandle MBundle = MembraneBundle->getBundle(fieldname);
      if (MBundle->isString("Name"))
      {
        StringHandle MName = MBundle->getString("Name");
        if (MName->get() == membranename)
        {
          error("A Membrane model cannot be used twice: CardioWave does not support this");
          return;
        }
      }
      membrane_num++;
      {
        std::ostringstream oss;
        oss << "Membrane_" << membrane_num;
        fieldname = oss.str(); 
      }
    }

    // Entry point for the converter library:
    SCIRunAlgo::ConverterAlgo mc(this);

    
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
    
    Membrane->setField("Geometry",Geometry);
   
    // Get parameters from GUI:
    std::string paramstr; 
    paramstr = guimembraneparam_.get();
    // Add the ones from the input port:
    if (Parameters_from_port.get_rep()) 
    {
      paramstr += "\n";
      paramstr += Parameters_from_port->get();
      paramstr += "\n";
    }


    Membrane->setString("Name",MembraneName);
    
    // Figure out how the membrane needs to be assigned this number
    // in cardiowave and add it to the parameters:
    SynapseItem item = synapsexml_.get_synapse(membranename);
    {
      std::ostringstream oss;
      oss << "\n" << item.nodetype << " = " << membrane_num << "\n";
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
    
    // Send the data downstream:
    send_output_handle("MembraneBundle",MembraneBundle,false);
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


