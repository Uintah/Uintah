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
  GuiInt    guiuseelements_;
  GuiInt    guiusefieldvalue_;
  GuiDouble guireferencevalue_;
  GuiDouble guireferencedomain_;
};


DECLARE_MAKER(DMDAddReference)

DMDAddReference::DMDAddReference(GuiContext* ctx)
  : Module("DMDAddReference", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
    guiusefieldvalue_(get_ctx()->subVar("usefieldvalue")),
    guireferencevalue_(get_ctx()->subVar("referencevalue")),
    guireferencedomain_(get_ctx()->subVar("referencedomain")),
    guiuseelements_(get_ctx()->subVar("useelements"))
{
}

void DMDAddReference::execute()
{
  // Make sure we have a reference bundle
  BundleHandle ReferenceBundle;
  FieldHandle  Geometry;
  MatrixHandle RefVal;
  MatrixHandle RefDomain;
  
  // required ones
  if (!(get_input_handle("Geometry",Geometry,true))) return;
  // optional ones
  get_input_handle("ReferenceBundle",ReferenceBundle,false);
  get_input_handle("Value",RefVal,false);
  get_input_handle("Domain",RefDomain,false);
  
  if (ReferenceBundle.get_rep() == 0) 
  {
    ReferenceBundle = scinew Bundle();
    if (ReferenceBundle.get_rep() == 0)
    {
      error("Could not allocate new reference bundle");
      return;
    } 
  } 
  
  // Try to find which reference numbers have already been taken
  int reference_num = 0;
  std::ostringstream oss;
  oss << "Reference_" << reference_num;
  while (ReferenceBundle->isBundle(oss.str()))
  {
    reference_num++;
    oss.clear();
    oss << "Reference_" << reference_num;
  }
  
  // Add a new bundle to the bundle with the data
  // from this module
  BundleHandle Reference = scinew Bundle();
  if (Reference.get_rep() == 0)
  {
    error("Could not allocate new reference bundle");
    return;
  }

  // Link new Bundle to main Bundle
  oss.clear();
  oss << "Reference_" << reference_num; 
  ReferenceBundle->setBundle(oss.str(),Reference);

  ModelCreation::ConverterAlgo mc(this);

  // fill out bundle with data
  Reference->setField("Geometry",Geometry);

  bool usefieldvalue = static_cast<bool>(guiusefieldvalue_.get());

  // If user wants to set the reference value explicitly
  if (usefieldvalue == false)
  {
    double referencevalue = 0.0;
    if (RefVal.get_rep())
    {
      mc.MatrixToDouble(RefVal,referencevalue);
    	guireferencevalue_.set(referencevalue);
    }
    referencevalue = guireferencevalue_.get();
    mc.DoubleToMatrix(referencevalue,RefVal);
    Reference->setMatrix("Value",RefVal);
  }

  double referencedomain = 0.0;
  if (RefDomain.get_rep())
  {
    mc.MatrixToDouble(RefDomain,referencedomain);
    guireferencedomain_.set(referencedomain);
  }
  referencedomain = guireferencedomain_.get();
  mc.DoubleToMatrix(referencedomain,RefDomain);
  Reference->setMatrix("Domain",RefDomain);

  bool useelement = static_cast<bool>(guiuseelements_.get());
  if (useelement)
  {
    MatrixHandle UseElement;
    mc.DoubleToMatrix(1.0,UseElement);
    Reference->setMatrix("UseElement",UseElement);
  }
  
  StringHandle SourceFile = scinew String("BCondZero.cc ");
  Reference->setString("SourceFile",SourceFile);

  StringHandle Parameters = scinew String("");
  Reference->setString("Parameters",Parameters);

  send_output_handle("ReferenceBundle",ReferenceBundle,true);
}



} // End namespace CardioWave


