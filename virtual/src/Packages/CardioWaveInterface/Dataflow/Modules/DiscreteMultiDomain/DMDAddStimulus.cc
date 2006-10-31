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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWaveInterface/Core/XML/SynapseXML.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWave {

using namespace SCIRun;

class DMDAddStimulus : public Module {
public:
  DMDAddStimulus(GuiContext*);
  virtual void execute();

private:  
  // TCL tools

  GuiDouble  guidomain_;
  GuiDouble  guicurrent_;
  GuiDouble  guistart_;
  GuiDouble  guiend_;
  GuiInt     guicurrentdensity_;
  GuiInt     guiuseelements_;
};


DECLARE_MAKER(DMDAddStimulus)

DMDAddStimulus::DMDAddStimulus(GuiContext* ctx)
  : Module("DMDAddStimulus", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guidomain_(get_ctx()->subVar("stim-domain")),
    guicurrent_(get_ctx()->subVar("stim-current")),
    guistart_(get_ctx()->subVar("stim-start")),
    guiend_(get_ctx()->subVar("stim-end")),
    guicurrentdensity_(get_ctx()->subVar("stim-is-current-density")),
    guiuseelements_(get_ctx()->subVar("stim-useelements"))
{
}

void DMDAddStimulus::execute()
{
  // Create input handles for objects:
  BundleHandle StimulusBundle;
  FieldHandle Geometry;
  MatrixHandle Domain;
  MatrixHandle Current;
  MatrixHandle Start;
  MatrixHandle End;
  MatrixHandle FieldDensity;

  // get the latest data
  if(!(get_input_handle("Geometry",Geometry,true))) return;
  get_input_handle("StimulusBundle",StimulusBundle,false);
  get_input_handle("Domain",Domain,false);
  get_input_handle("Current",Current,false);
  get_input_handle("Start",Start,false);
  get_input_handle("End",End,false);

  // If nothing changed, do nothing:
  if (inputs_changed_ || guidomain_.changed() || guicurrent_.changed() || 
      guistart_.changed() || guiend_.changed() || guicurrentdensity_.changed() || 
      guiuseelements_.changed() || !oport_cached("StimulationBundle"))
  {
    // Add entrypoint to converter library
    SCIRunAlgo::ConverterAlgo mc(this);
    double val;

    if (mc.MatrixToDouble(Domain,val)) guidomain_.set(val);
    if (mc.MatrixToDouble(Current,val)) guicurrent_.set(val);
    if (mc.MatrixToDouble(Start,val)) guistart_.set(val);      
    if (mc.MatrixToDouble(End,val)) guiend_.set(val);  
    
    if (StimulusBundle.get_rep() == 0)
    {
      StimulusBundle = scinew Bundle();
      if (StimulusBundle.get_rep() == 0)
      {
        error("Could not allocate new stimulus bundle");
        return;
      }
    }
    else
    {
      StimulusBundle.detach();
    }

    
    int stimulus_num = 0;
    std::string fieldname;
    
    {
      std::ostringstream oss;
      oss << "Stimulus_" << stimulus_num;
      fieldname = oss.str(); 
    }
    
    while (StimulusBundle->isBundle(fieldname))
    {
      stimulus_num++;
      {
        std::ostringstream oss;
        oss << "Stimulus_" << stimulus_num;
        fieldname = oss.str(); 
      }
    }

    if (stimulus_num > 0)
    {
      std::string oldsourcefile;    
      BundleHandle OldStimulus;
      StringHandle OldSourceFile;
      
      if (StimulusBundle->isBundle("Stimulus_0"))
      {
        OldStimulus = StimulusBundle->getBundle("Stimulus_0");
        OldSourceFile = OldStimulus->getString("SourceFile");
        if (OldSourceFile.get_rep())
        {
          oldsourcefile = OldSourceFile->get();
          if (oldsourcefile != "StimFile.c ")
          {
            error("CardioWave does not allow for different stimulus models to be mixed together");
            return;
          }
        }  
      }
    }

    BundleHandle Stimulus;
    Stimulus = scinew Bundle();
    if (Stimulus.get_rep() == 0)
    {
      error("Could not allocate new stimulus bundle");
      return;
    }
    
    {
      std::ostringstream oss;
      oss << "Stimulus_" << stimulus_num;
      fieldname = oss.str(); 
    }
    StimulusBundle->setBundle(fieldname,Stimulus);
      
    Stimulus->setField("Geometry",Geometry);

    val = guidomain_.get(); mc.DoubleToMatrix(val,Domain); 
    val = guidomain_.get(); mc.DoubleToMatrix(val,Current); 
    val = guistart_.get(); mc.DoubleToMatrix(val,Start); 
    val = guiend_.get(); mc.DoubleToMatrix(val,End); 
   
    int fielddensity = guicurrentdensity_.get();
    mc.IntToMatrix(fielddensity,FieldDensity); 

    MatrixHandle UseElements;
    int useelements = guiuseelements_.get();
    mc.IntToMatrix(useelements,UseElements);

    Stimulus->setMatrix("Domain",Domain);
    Stimulus->setMatrix("Current",Current);
    Stimulus->setMatrix("Start",Start);
    Stimulus->setMatrix("End",End);
    Stimulus->setMatrix("FieldDensity",FieldDensity);
    Stimulus->setMatrix("UseElements",UseElements);

    StringHandle SourceFile = scinew String("StimFile.c ");
    Stimulus->setString("SourceFile",SourceFile);

    StringHandle Parameters = scinew String("");
    Stimulus->setString("Parameters",Parameters);

    // Send new data downstream:
    send_output_handle("StimulusBundle",StimulusBundle,true);
  }
}

} // End namespace CardioWave

