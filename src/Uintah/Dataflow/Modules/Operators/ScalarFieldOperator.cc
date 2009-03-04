/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#include "ScalarFieldOperator.h"

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include <cmath>


using namespace SCIRun;

namespace Uintah {
 
class ScalarFieldOperator: public Module
{
public:
  
  ScalarFieldOperator(GuiContext* ctx);
  virtual ~ScalarFieldOperator() {}
    
  virtual void execute(void);
    
private:
  //    TCLstring tcl_status;
  GuiInt guiOperation;

  FieldIPort *in;

  FieldOPort *sfout;
  //ScalarFieldOPort *vfout;
};

} // end namespace Uintah

using namespace Uintah;

DECLARE_MAKER(ScalarFieldOperator)

ScalarFieldOperator::ScalarFieldOperator(GuiContext* ctx)
  : Module("ScalarFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(get_ctx()->subVar("operation"))
{
}

  
void ScalarFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Scalar Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  if(!in->get(hTF) ){
    error("execute(void) Didn't get a handle.");
    return;
  } else if ( !hTF->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field.");
    return;
  }

  
  const SCIRun::TypeDescription *ftd = hTF->get_type_description();
  CompileInfoHandle ci = ScalarFieldOperatorAlgo::get_compile_info(ftd);
  Handle<ScalarFieldOperatorAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }

  FieldHandle fh =  algo->execute( hTF, guiOperation.get() );
  if( fh.get_rep() != 0 ){
    sfout->send(fh);
  }
}


CompileInfoHandle
ScalarFieldOperatorAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarFieldOperatorAlgoT");
  static const string base_class_name("ScalarFieldOperatorAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}




