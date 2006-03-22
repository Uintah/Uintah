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
#include <Dataflow/Ports/MatrixPort.h>
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

class DMDAddBlockStimulus : public Module {
public:
  DMDAddBlockStimulus(GuiContext*);
  virtual void execute();

private:  
  // TCL tools

  GuiDouble  guidomain_;
  GuiDouble  guicurrent_;
  GuiDouble  guistart_;
  GuiDouble  guiend_;
  GuiInt     guicurrentdensity_;
};


DECLARE_MAKER(DMDAddBlockStimulus)

DMDAddBlockStimulus::DMDAddBlockStimulus(GuiContext* ctx)
  : Module("DMDAddBlockStimulus", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guidomain_(ctx->subVar("stim-domain")),
    guicurrent_(ctx->subVar("stim-current")),
    guistart_(ctx->subVar("stim-start")),
    guiend_(ctx->subVar("stim-end")),
    guicurrentdensity_(ctx->subVar("stim-is-current-density"))
{
}

void DMDAddBlockStimulus::execute()
{
  ModelCreation::ConverterAlgo mc(this);

  BundleIPort* stimulusbundle_iport = dynamic_cast<BundleIPort*>(getIPort(0));
  if (stimulusbundle_iport == 0)
  {
    error("Could not find stimulus bundle input port");
    return;
  }
  
  int stimulus_num = 0;
  
  BundleHandle StimulusBundle;
  BundleHandle Stimulus;
  
  if (stimulusbundle_iport->get(StimulusBundle))
  {
    // In case we already have a few other stimuli lined up
    
    // Determine the nodetype numbers already used.

    std::ostringstream oss;
    oss << "stimulus_" << stimulus_num;
    while (StimulusBundle->isBundle(oss.str()))
    {
      stimulus_num++;
      oss.clear();
      oss << "stimulus_" << stimulus_num;
    }
  }
  else
  {
    // Create a new output bundle
  
    StimulusBundle = scinew Bundle();
    if (StimulusBundle.get_rep() == 0)
    {
      error("Could not allocate new stimulus bundle");
      return;
    } 
  }

  // Add a new bundle to the bundle with the data
  // from this module
  Stimulus = scinew Bundle();
  if (Stimulus.get_rep() == 0)
  {
    error("Could not allocate new stimulus bundle");
    return;
  }
  
  std::ostringstream oss;
  oss << "stimulus_" << stimulus_num; 
  StimulusBundle->setBundle(oss.str(),Stimulus);
    
  FieldIPort* geometryport = dynamic_cast<FieldIPort*>(getIPort(1));
  if (geometryport == 0)
  {
    error("Could not find Stimulus Geometry port");
    return;
  }

  FieldHandle Geometry;
  geometryport->get(Geometry);
  
  if (Geometry.get_rep() == 0)
  {
    error("Stimulus Geometry field is empty");
    return;  
  }

  Stimulus->setField("field",Geometry);


  MatrixHandle Domain;
  MatrixHandle Current;
  MatrixHandle Start;
  MatrixHandle End;
  MatrixHandle FieldDensity;
  MatrixIPort* matrix_port;

  double val;
  
  if (matrix_port = dynamic_cast<MatrixIPort*>(getIPort(1)))
  {
    error("Could not find Stimulus Domain port");
    return;
  }

  matrix_port->get(Domain);
  if (Domain.get_rep())
  {
    if (mc.MatrixToDouble(Domain,val))
    {
      guidomain_.set(val);
    }    
  }

  val = guidomain_.get();
  mc.DoubleToMatrix(val,Domain); 


  if (matrix_port = dynamic_cast<MatrixIPort*>(getIPort(2)))
  {
    error("Could not find Stimulus Current port");
    return;
  }

  matrix_port->get(Current);
  if (Current.get_rep())
  {
    if (mc.MatrixToDouble(Current,val))
    {
      guidomain_.set(val);
    }    
  }

  val = guidomain_.get();
  mc.DoubleToMatrix(val,Current); 
 
  if (matrix_port = dynamic_cast<MatrixIPort*>(getIPort(3)))
  {
    error("Could not find Stimulus Start port");
    return;
  }

  matrix_port->get(Current);
  if (Start.get_rep())
  {
    if (mc.MatrixToDouble(Start,val))
    {
      guistart_.set(val);
    }    
  }

  val = guistart_.get();
  mc.DoubleToMatrix(val,Start); 
 
  if (matrix_port = dynamic_cast<MatrixIPort*>(getIPort(4)))
  {
    error("Could not find Stimulus End port");
    return;
  }

  matrix_port->get(End);
  if (End.get_rep())
  {
    if (mc.MatrixToDouble(End,val))
    {
      guiend_.set(val);
    }    
  }

  val = guiend_.get();
  mc.DoubleToMatrix(val,End); 
 
  int fielddensity;
  fielddensity = guicurrentdensity_.get();
  mc.IntToMatrix(fielddensity,FieldDensity); 

  Stimulus->setMatrix("domain",Domain);
  Stimulus->setMatrix("current",Current);
  Stimulus->setMatrix("start",Start);
  Stimulus->setMatrix("end",End);
  Stimulus->setMatrix("fielddensity",FieldDensity);

  StringHandle SourceFile = scinew String("StimFile.cc ");
  Stimulus->setString("sourcefile",SourceFile);

  StringHandle Parameters = scinew String("");
  Stimulus->setString("parameters",Parameters);


  if (StimulusBundle.get_rep())
  {
    BundleOPort* oport = dynamic_cast<BundleOPort*>(getOPort(0));
    if (oport)
    {
      oport->send(StimulusBundle);
    }  
  }
}

} // End namespace CardioWave

