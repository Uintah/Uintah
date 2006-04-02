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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/CardioWave/Core/XML/SynapseXML.h>
#include <Packages/ModelCreation/Core/Converter/ConverterAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h> 

namespace CardioWave {

using namespace SCIRun;

class DMDAddReference : public Module {
public:
  DMDAddReference(GuiContext*);
  virtual void execute();

private:  
  // TCL tools
  GuiInt    guiusefieldvalue_;
  GuiDouble guireferencevalue_;
};


DECLARE_MAKER(DMDAddReference)

DMDAddReference::DMDAddReference(GuiContext* ctx)
  : Module("DMDAddReference", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guiusefieldvalue_(get_ctx()->subVar("usefieldvalue")),
    guireferencevalue_(get_ctx()->subVar("referencevalue"))
{
}

void DMDAddReference::execute()
{
  ModelCreation::ConverterAlgo mc(this);

  BundleIPort* referencebundle_iport = dynamic_cast<BundleIPort*>(get_input_port(0));
  if (referencebundle_iport == 0)
  {
    error("Could not find reference bundle input port");
    return;
  }
  
  int reference_num = 0;
  
  BundleHandle ReferenceBundle;
  BundleHandle Reference;
  
  if (referencebundle_iport->get(ReferenceBundle))
  {
    // In case we already have a few other membranes lined up
    
    // Determine the nodetype numbers already used.

    std::ostringstream oss;
    oss << "reference_" << reference_num;
    while (ReferenceBundle->isBundle(oss.str()))
    {
      reference_num++;
      oss.clear();
      oss << "reference_" << reference_num;
    }
  }
  else
  {
    // Create a new output bundle
  
    ReferenceBundle = scinew Bundle();
    if (ReferenceBundle.get_rep() == 0)
    {
      error("Could not allocate new reference bundle");
      return;
    } 
  }

  // Add a new bundle to the bundle with the data
  // from this module
  Reference = scinew Bundle();
  if (Reference.get_rep() == 0)
  {
    error("Could not allocate new reference bundle");
    return;
  }
  
  std::ostringstream oss;
  oss << "reference_" << reference_num; 
  ReferenceBundle->setBundle(oss.str(),Reference);
    
  FieldIPort* geometryport = dynamic_cast<FieldIPort*>(get_input_port(1));
  if (geometryport == 0)
  {
    error("Could not find Reference Geometry port");
    return;
  }

  FieldHandle Geometry;
  geometryport->get(Geometry);
  
  if (Geometry.get_rep() == 0)
  {
    error("Reference Geometry field is empty");
    return;  
  }

  Reference->setField("field",Geometry);
 
  bool usefieldvalue = guiusefieldvalue_.get();

  if (usefieldvalue == true)
  {
    MatrixIPort* referenceval_port = dynamic_cast<MatrixIPort*>(get_input_port(2));
    if (referenceval_port == 0)
    {
      error("Could not find Reference Value input port");
      return;
    } 

    MatrixHandle RefVal;
    referenceval_port->get(RefVal);  

    double referencevalue = 0.0;
    if (RefVal.get_rep())
    {
      mc.MatrixToDouble(RefVal,referencevalue);
    	guireferencevalue_.set(referencevalue);
    }

    referencevalue = guireferencevalue_.get();
    mc.DoubleToMatrix(referencevalue,RefVal);

    Reference->setMatrix("value",RefVal);
  }
  
  
  StringHandle SourceFile = scinew String("BCondZero.cc ");
  Reference->setString("sourcefile",SourceFile);

  StringHandle Parameters = scinew String("");
  Reference->setString("parameters",Parameters);

  if (ReferenceBundle.get_rep())
  {
    BundleOPort* oport = dynamic_cast<BundleOPort*>(get_output_port(0));
    if (oport)
    {
      oport->send(ReferenceBundle);
    }  
  }
}



} // End namespace CardioWave


