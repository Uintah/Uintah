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
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

namespace CardioWave {

using namespace SCIRun;

class DMDCreateDomain : public Module {
public:
  DMDCreateDomain(GuiContext*);
  virtual void execute();

};

DECLARE_MAKER(DMDCreateDomain)

DMDCreateDomain::DMDCreateDomain(GuiContext* ctx)
  : Module("DMDCreateDomain", ctx, Source, "DiscreteMultiDomain", "CardioWave")
{
}

void DMDCreateDomain::execute()
{
  // Define input handles:
  FieldHandle Conductivity;
  FieldHandle ElementType;
  FieldHandle InitialPotential;
  MatrixHandle ConductivityTable;
  MatrixHandle NodeLink;
  MatrixHandle ElemLink;
  
  // Get the latest input objects:
  if (!(get_input_handle("Conductivity",Conductivity,true))) return;
  if (!(get_input_handle("ElementType",ElementType,true))) return;
  get_input_handle("Initial Potential",InitialPotential,false);
  get_input_handle("ConductivityTable",ConductivityTable,false); 

  // A module should only execute when data has changed otherwise it should return
  // the cached values:
  if (inputs_changed_ || !oport_cached("DomainBundle"))
  {

    SCIRunAlgo::FieldsAlgo falgo(this);

    if (InitialPotential.get_rep())
    {
      if (InitialPotential->basis_order() == 0) 
      {
        falgo.FieldDataElemToNode(InitialPotential,InitialPotential,"average");
      }
      else if (InitialPotential->basis_order() == -1)
      {
        error("The initial potential field does not have any values assigned to it");
        return;
      }
    }
    
    if(ElementType->is_property("NodeLink")) ElementType->get_property("NodeLink",NodeLink);
    if(ElementType->is_property("ElemLink")) ElementType->get_property("ElemLink",ElemLink);

    // Create the output object:
    BundleHandle output = scinew Bundle();
    if (output.get_rep() == 0)
    {
      error("Could not allocate output Bundle");
      return;
    }

    // Add all the input to the output object:
    output->setField("Conductivity",Conductivity);
    output->setField("ElementType",ElementType);
    output->setMatrix("ConductivityTable",ConductivityTable);
    output->setMatrix("NodeLink",NodeLink);
    output->setMatrix("ElemLink",ElemLink);
    if (InitialPotential.get_rep()) output->setField("InitialPotential",InitialPotential);
    
    std::string sourcefile = "DomainSPRfile.c ";
    StringHandle SourceFile = scinew String(sourcefile);
    if (SourceFile.get_rep() == 0)
    {
      error("Could not allocate String");
      return;
    }
    output->setString("SourceFile",SourceFile);
    
    std::string parameters = "scale_int=1.0\nscale_ext=1.0\nscale_bath=1.0\nscale_area=1.0\n";
    StringHandle Parameters = scinew String(parameters);
    if (Parameters.get_rep() == 0)
    {
      error("Could not allocate String");
      return;
    }
  
    output->setString("Parameters",Parameters);
    
    // Wrap the bundle in a new bundle so we can merge everything together
    // downstream:
    BundleHandle DomainBundle = scinew Bundle;
    if (DomainBundle.get_rep() == 0)
    {
      error("Could not allocate DomainBundle");
      return;
    }

    DomainBundle->setBundle("Domain",output);

    // Send the output:
    send_output_handle("DomainBundle",DomainBundle,false);
  }
}

} // End namespace CardioWave


