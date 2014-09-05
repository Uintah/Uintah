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
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWaveInterface {

using namespace SCIRun;

class DMDAddMembraneElectrodes : public Module {
public:
  DMDAddMembraneElectrodes(GuiContext*);
  virtual void execute();

};


DECLARE_MAKER(DMDAddMembraneElectrodes)
DMDAddMembraneElectrodes::DMDAddMembraneElectrodes(GuiContext* ctx)
  : Module("DMDAddMembraneElectrodes", ctx, Source, "DiscreteMultiDomain", "CardioWaveInterface")
{
}

void DMDAddMembraneElectrodes::execute()
{
  // Define input handles:
  BundleHandle ElectrodeBundle;
  FieldHandle  Electrodes;
  FieldHandle  Membrane;

  // Get the input from the ports:
  if (!(get_input_handle("Electrodes",Electrodes,true))) return;
  get_input_handle("ElectrodeBundle",ElectrodeBundle,false);
  if(!(get_input_handle("Membrane",Membrane,false))) return;

 // Test whether we need to execute:
  if (inputs_changed_ || !oport_cached("ElectrodeBundle"))
  {
    // Test whether we have a ReferenceBundle on the input:
    if (ElectrodeBundle.get_rep() == 0) 
    {
      ElectrodeBundle = scinew Bundle();
      if (ElectrodeBundle.get_rep() == 0)
      {
        error("Could not allocate new electrode bundle");
        return;
      } 
    } 
    else
    {
      // We need a copy any way to conform to the dataflow system:
      ElectrodeBundle.detach();
    }
    
    // Try to find which reference numbers have already been taken:
    int electrode_num = 0;
    std::string fieldname;
    
    {
      std::ostringstream oss;
      oss << "Electrode_" << electrode_num;
      fieldname = oss.str(); 
    }
    while (ElectrodeBundle->isBundle(fieldname))
    {
      electrode_num++;
      {
        std::ostringstream oss;
        oss << "Electrode_" << electrode_num;
        fieldname = oss.str(); 
      }
    }
    
    // Add a new bundle to the bundle with the data
    // from this module
    BundleHandle Electrode = scinew Bundle();
    if (Electrode.get_rep() == 0)
    {
      error("Could not allocate new electrode bundle");
      return;
    }

    // Link new Bundle to main Bundle
    {
      std::ostringstream oss;
      oss << "Electrode_" << electrode_num;
      fieldname = oss.str(); 
    }
    ElectrodeBundle->setBundle(fieldname,Electrode);

    // Get entry point to converter library
    SCIRunAlgo::ConverterAlgo mc(this);
    SCIRunAlgo::FieldsAlgo falgo(this);

    if (!(falgo.ClearAndChangeFieldBasis(Membrane,Membrane,"Linear")))
    {
      error("DMDAddMembraneElectrodes: Could not build a linear field for the membrane");
      return;        
    }

    // fill out bundle with data
    Electrode->setField("Electrodes",Electrodes);
    Electrode->setField("Membrane",Membrane);

    
    StringHandle SourceFile = scinew String("OutputSCIRun.cc ");
    Electrode->setString("SourceFile",SourceFile);

    StringHandle Parameters = scinew String("scirun_dump_electrode=yes\nscirun_dump_electrode_time=yes\n");
    Electrode->setString("Parameters",Parameters);

    // Send data downstream:
    send_output_handle("ElectrodeBundle",ElectrodeBundle,false);
  }
}

} // End namespace CardioWave


