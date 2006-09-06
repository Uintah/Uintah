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

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/String.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWave/Core/XML/SynapseXML.h>
#include <Packages/CardioWave/Core/XML/NWSolverXML.h>
#include <Packages/CardioWave/Core/XML/OutputXML.h>
#include <Packages/CardioWave/Core/XML/NeuroWaveXML.h>
#include <Packages/CardioWave/Core/XML/NWTimeStepXML.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWave {

using namespace SCIRun;

class DMDCreateSimulation : public Module {
public:
  DMDCreateSimulation(GuiContext*);

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:  
  // TCL tools
  std::string totclstring(std::string &instring);
  std::string convertclist(std::vector<std::string> list);

  GuiString guisolvernames_;
  GuiString guisolvername_;
  GuiString guisolverparam_;
  GuiString guisolverdesc_;

  GuiString guitstepnames_;
  GuiString guitstepname_;
  GuiString guitstepparam_;
  GuiString guitstepdesc_;

  GuiString guioutputnames_;
  GuiString guioutputname_;
  GuiString guioutputparam_;
  GuiString guioutputdesc_;

  GuiString guicwaveparam_;
  GuiString guicwavedesc_;

  NWSolverXML    solverxml_;  
  NWTimeStepXML  tstepxml_;  
  OutputXML      outputxml_;    
  NeuroWaveXML   neurowavexml_;  
  
};


DECLARE_MAKER(DMDCreateSimulation)
DMDCreateSimulation::DMDCreateSimulation(GuiContext* ctx)
  : Module("DMDCreateSimulation", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guisolvernames_(get_ctx()->subVar("solver-names")),
    guisolvername_(get_ctx()->subVar("solver-name")),
    guisolverparam_(get_ctx()->subVar("solver-param")),
    guisolverdesc_(get_ctx()->subVar("solver-desc")),
    guitstepnames_(get_ctx()->subVar("tstep-names")),
    guitstepname_(get_ctx()->subVar("tstep-name")),
    guitstepparam_(get_ctx()->subVar("tstep-param")),
    guitstepdesc_(get_ctx()->subVar("tstep-desc")),
    guioutputnames_(get_ctx()->subVar("output-names")),
    guioutputname_(get_ctx()->subVar("output-name")),
    guioutputparam_(get_ctx()->subVar("output-param")),
    guioutputdesc_(get_ctx()->subVar("output-desc")),
    guicwaveparam_(get_ctx()->subVar("cwave-param")),
    guicwavedesc_(get_ctx()->subVar("cwave-desc"))
{
    std::string defaultname;
    std::string param;
    std::string desc;
    defaultname = solverxml_.get_default_name();
    guisolvername_.set(defaultname);
    NWSolverItem solveritem = solverxml_.get_nwsolver(defaultname);
    param = solveritem.parameters;
    desc  = solveritem.description;
    guisolverparam_.set(param);
    guisolverdesc_.set(desc);

    defaultname = tstepxml_.get_default_name();
    guitstepname_.set(defaultname);
    NWTimeStepItem tstepitem = tstepxml_.get_nwtimestep(defaultname);
    param = tstepitem.parameters;
    desc  = tstepitem.description;
    guitstepparam_.set(param);
    guitstepdesc_.set(desc);
 
    defaultname = outputxml_.get_default_name();
    guioutputname_.set(defaultname);
    OutputItem outputitem = outputxml_.get_output(defaultname);
    param = outputitem.parameters;
    desc  = outputitem.description;
    guioutputparam_.set(param);
    guioutputdesc_.set(desc);
    
    NeuroWaveItem cwaveitem = neurowavexml_.get_neurowave();
    param = cwaveitem.parameters;
    desc  = cwaveitem.description;
    guicwaveparam_.set(param);
    guicwavedesc_.set(desc);
    
    ctx->reset();    
}


void DMDCreateSimulation::execute()
{
  // Define input handles:
  BundleHandle SimulationBundle;
  BundleHandle MembraneBundle;
  BundleHandle DomainBundle;
  BundleHandle StimulusBundle;
  BundleHandle ReferenceBundle;
  BundleHandle ElectrodeBundle;
  StringHandle ExtParameters;
  
  // required ones
  if (!(get_input_handle("DomainBundle",DomainBundle,true))) return;
  if (!(get_input_handle("MembraneBundle",MembraneBundle,true))) return;
  if (!(get_input_handle("StimulusBundle",StimulusBundle,true))) return;
  if (!(get_input_handle("ReferenceBundle",ReferenceBundle,true))) return;

  get_input_handle("ElectrodeBundle",ElectrodeBundle,false);
  get_input_handle("Parameters",ExtParameters,false);
  

  // get all input from GUI 
  std::string cmd;
  cmd = get_id() + " get_param";
  get_gui()->lock();
  get_gui()->eval(cmd);
  get_gui()->unlock();
  get_ctx()->reset();

  if (inputs_changed_  || guisolvername_.changed() || guisolverparam_.changed() || 
      guitstepname_.changed() || guitstepparam_.changed() || guioutputname_.changed() || 
      guioutputparam_.changed() || guicwaveparam_.changed() || !oport_cached("SimulationBundle") )
  {

    SimulationBundle = scinew Bundle();
    if (SimulationBundle.get_rep() == 0)
    {
      error("Could not allocate new simulation bundle");
      return;
    }   

    // start merging everything together
    SimulationBundle->merge(DomainBundle);
    SimulationBundle->merge(MembraneBundle);
    SimulationBundle->merge(StimulusBundle);
    SimulationBundle->merge(ReferenceBundle);
    if (ElectrodeBundle.get_rep()) SimulationBundle->merge(ElectrodeBundle);

    std::string solvername = guisolvername_.get();
    std::string tstepname  = guitstepname_.get();
    std::string outputname = guioutputname_.get();
    
    NWSolverItem solveritem = solverxml_.get_nwsolver(solvername);
    OutputItem outputitem = outputxml_.get_output(outputname);
    NWTimeStepItem tstepitem = tstepxml_.get_nwtimestep(tstepname);
      
    std::string paramstr = guisolverparam_.get();
    paramstr += "# Parameters for timestepper\n\n";
    paramstr += guitstepparam_.get();
    paramstr += "# Parameters for output module\n\n";
    paramstr += guioutputparam_.get();
    paramstr += "# Parameters for solver\n\n";
    paramstr += guicwaveparam_.get();

    DomainBundle = SimulationBundle->getBundle("Domain");
    if (DomainBundle.get_rep() == 0)
    {
      error("DomainBundle does not contain a sub bundle called Domain");
      return;
    }
    StringHandle DomainParam = DomainBundle->getString("Parameters");
    if (DomainParam.get_rep())
    {
      paramstr += "# Parameters for domain\n\n";
      paramstr += DomainParam->get() + "\n";
    }
    
    int membrane_num;
    int reference_num;
    int stimulus_num;
    int electrode_num;
    std::string fieldname;
    membrane_num = 1;

    {
      std::ostringstream oss;
      oss << "Membrane_" << membrane_num;
      fieldname = oss.str();
    }
    
    while (SimulationBundle->isBundle(fieldname))
    {
      MembraneBundle = SimulationBundle->getBundle(fieldname);
      StringHandle MembraneParam = MembraneBundle->getString("Parameters");
      if (MembraneParam.get_rep())
      {
        std::ostringstream oss;
        oss << membrane_num;
        paramstr += "# Parameters for membrane " + oss.str() + "\n\n";
        paramstr += MembraneParam->get() + "\n";
      }
      membrane_num++;
      {
        std::ostringstream oss;
        oss << "Membrane_" << membrane_num;
        fieldname = oss.str();
      }
    }


    stimulus_num = 0;

    {
      std::ostringstream oss;
      oss << "Stimulus_" << stimulus_num;
      fieldname = oss.str();
    }

    while (SimulationBundle->isBundle(fieldname))
    {
      StimulusBundle = SimulationBundle->getBundle(fieldname);
      StringHandle StimulusParam = StimulusBundle->getString("Parameters");
      if (StimulusParam.get_rep())
      {
        std::ostringstream oss;
        oss << stimulus_num;
        paramstr += "# Parameters for stimulus " + oss.str() + "\n\n";
        paramstr += StimulusParam->get() + "\n";
      }
      stimulus_num++;
      {
        std::ostringstream oss;
        oss << "Stimulus_" << stimulus_num;
        fieldname = oss.str();
      }
    }

    reference_num = 0;
    {
      std::ostringstream oss;
      oss << "Reference_" << reference_num;
      fieldname = oss.str();
    }
    while (SimulationBundle->isBundle(fieldname))
    {
      ReferenceBundle = SimulationBundle->getBundle(fieldname);
      StringHandle ReferenceParam = ReferenceBundle->getString("Parameters");
      if (ReferenceParam.get_rep())
      {
        std::ostringstream oss;
        oss << reference_num;
        paramstr += "# Parameters for reference " + oss.str() + "\n\n";
        paramstr += ReferenceParam->get() + "\n";
      }
      reference_num++;
      {
        std::ostringstream oss;
        oss << "Reference_" << reference_num;
        fieldname = oss.str();
      }
    }

    electrode_num = 0;
    {
      std::ostringstream oss;
      oss << "Electrode_" << electrode_num;
      fieldname = oss.str();
    }
    while (SimulationBundle->isBundle(fieldname))
    {
      ElectrodeBundle = SimulationBundle->getBundle(fieldname);
      StringHandle ElectrodeParam = ElectrodeBundle->getString("Parameters");
      if (ElectrodeParam.get_rep())
      {
        std::ostringstream oss;
        oss << electrode_num;
        paramstr += "# Parameters for electrode " + oss.str() + "\n\n";
        paramstr += ElectrodeParam->get() + "\n";
      }
      electrode_num++;
      {
        std::ostringstream oss;
        oss << "Electrode_" << electrode_num;
        fieldname = oss.str();
      }
    }

    if (ExtParameters.get_rep())
    {
      paramstr += "# Parameters from module input port\n";
      paramstr += ExtParameters->get() + "\n";
    }

    StringHandle Parameters = scinew String(paramstr);
    if (Parameters.get_rep() == 0)
    {
      error("Could not create parameter string");
      return;
    } 
    SimulationBundle->setString("Parameters",Parameters);
       
       
    std::string sourcefiles = "NeuroKernel.c VectorOps.c CWaveKernel.c ";
    sourcefiles += solveritem.file + " ";  
    sourcefiles += tstepitem.file + " ";  
    sourcefiles += outputitem.file + " ";  

    DomainBundle = SimulationBundle->getBundle("Domain");
    if (DomainBundle.get_rep() == 0)
    {
      error("DomainBundle does not contain a sub bundle called Domain");
      return;
    }
    StringHandle DomainSourceFile = DomainBundle->getString("SourceFile");
    if (DomainSourceFile.get_rep())
    {
      sourcefiles += DomainSourceFile->get() + " ";
    }

    membrane_num = 0;
    {
      std::ostringstream oss;
      oss << "Membrane_" << membrane_num;
      fieldname = oss.str();
    }
    
    while (SimulationBundle->isBundle(fieldname))
    {
      MembraneBundle = SimulationBundle->getBundle(fieldname);
      StringHandle MembraneSourceFile = MembraneBundle->getString("SourceFile");
      if (MembraneSourceFile.get_rep())
      {
        sourcefiles += MembraneSourceFile->get() + " ";
      }
      membrane_num++;
      {
        std::ostringstream oss;
        oss << "Membrane_" << membrane_num;
        fieldname = oss.str();
      }  
    }

    stimulus_num = 0;
    {
      std::ostringstream oss;
      oss << "Stimulus_" << stimulus_num;
      fieldname = oss.str();
    }
      
    while (SimulationBundle->isBundle(fieldname))
    {
      StimulusBundle = SimulationBundle->getBundle(fieldname);
      StringHandle StimulusSourceFile = StimulusBundle->getString("SourceFile");
      if (StimulusSourceFile.get_rep())
      {
        sourcefiles += StimulusSourceFile->get() + " ";
      }
      stimulus_num++;
      {
        std::ostringstream oss;
        oss << "Stimulus_" << stimulus_num;
        fieldname = oss.str();
      }
    }

    reference_num = 0;
    {
      std::ostringstream oss;
      oss << "Reference_" << reference_num;
      fieldname = oss.str();
    }
    while (SimulationBundle->isBundle(fieldname))
    {
      ReferenceBundle = SimulationBundle->getBundle(fieldname);
      StringHandle ReferenceSourceFile = ReferenceBundle->getString("SourceFile");
      if (ReferenceSourceFile.get_rep())
      {
        sourcefiles += ReferenceSourceFile->get() + " ";
      }
      reference_num++;
      {
        std::ostringstream oss;
        oss << "Reference_" << reference_num;
        fieldname = oss.str();
      }
    }

    electrode_num = 0;
    {
      std::ostringstream oss;
      oss << "Electrode_" << electrode_num;
      fieldname = oss.str();
    }
    while (SimulationBundle->isBundle(fieldname))
    {
      ElectrodeBundle = SimulationBundle->getBundle(fieldname);
      StringHandle ElectrodeSourceFile = ElectrodeBundle->getString("SourceFile");
      if (ElectrodeSourceFile.get_rep())
      {
        sourcefiles += ElectrodeSourceFile->get() + " ";
      }
      electrode_num++;
      {
        std::ostringstream oss;
        oss << "Electrode_" << electrode_num;
        fieldname = oss.str();
      }
    }



    
    StringHandle SourceFile = scinew String(sourcefiles);
    SimulationBundle->setString("SourceFile",SourceFile);
    send_output_handle("SimulationBundle",SimulationBundle,false);
  }
}

void
 DMDCreateSimulation::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() > 1)
  {
    if (args[1] == "get_solver_names")
    {
      std::vector<std::string> names = solverxml_.get_names(); 
      guisolvernames_.set(convertclist(names));
    }
    else if (args[1] == "set_solver")
    {
      if (args[2] != "")
      {
        NWSolverItem item = solverxml_.get_nwsolver(args[2]);
        std::string param = item.parameters;
        std::string desc  = item.description;
        guisolvername_.set(args[2]);
        guisolverparam_.set(param);
        guisolverdesc_.set(desc);
        get_ctx()->reset();
      }
    }
    else if (args[1] == "get_tstep_names")
    {
      std::vector<std::string> names = tstepxml_.get_names(); 
      guitstepnames_.set(convertclist(names));
    }
    else if (args[1] == "set_tstep")
    {
      if (args[2] != "")
      {
        NWTimeStepItem item = tstepxml_.get_nwtimestep(args[2]);
        std::string param = item.parameters;
        std::string desc  = item.description;
        guitstepname_.set(args[2]);
        guitstepparam_.set(param);
        guitstepdesc_.set(desc);
        get_ctx()->reset();
      }
    }
    else if (args[1] == "get_output_names")
    {
      std::vector<std::string> names = outputxml_.get_names(); 
      guioutputnames_.set(convertclist(names));
    }
    else if (args[1] == "set_output")
    {
      if (args[2] != "")
      {
        OutputItem item = outputxml_.get_output(args[2]);
        std::string param = item.parameters;
        std::string desc  = item.description;
        guioutputname_.set(args[2]);
        guioutputparam_.set(param);
        guioutputdesc_.set(desc);
        get_ctx()->reset();
      }
    }
    else if (args[1] == "set_cwave")
    {
      NeuroWaveItem item = neurowavexml_.get_neurowave();
      std::string param = item.parameters;
      guicwaveparam_.set(param);
      std::string desc = item.description;
      guicwavedesc_.set(desc);
      get_ctx()->reset();
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


std::string DMDCreateSimulation::totclstring(std::string &instring)
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

std::string DMDCreateSimulation::convertclist(std::vector<std::string> list)
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


