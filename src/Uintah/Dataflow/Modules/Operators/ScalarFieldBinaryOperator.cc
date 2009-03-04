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


#include "ScalarFieldBinaryOperator.h"

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Uintah/Core/Disclosure/TypeUtils.h>

#include <cmath>

using namespace std;

//#include <SCICore/Math/Mat.h>

#define TYPES_MUST_MATCH 1

using namespace SCIRun;

namespace Uintah {
class ScalarFieldBinaryOperator: public Module
{
public:

  ScalarFieldBinaryOperator(GuiContext* ctx);
  virtual ~ScalarFieldBinaryOperator() {}
    
  virtual void execute(void);
    
protected:
    
private:
    //    TCLstring tcl_status;
  GuiInt guiOperation;
  
  FieldIPort *in_left;
  FieldIPort *in_right;
  
  FieldOPort *sfout;
  //ScalarFieldOPort *vfout;
};

}// end namespace Uintah

using namespace Uintah;
 
DECLARE_MAKER(ScalarFieldBinaryOperator)

ScalarFieldBinaryOperator::ScalarFieldBinaryOperator(GuiContext* ctx)
  : Module("ScalarFieldBinaryOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(get_ctx()->subVar("operation"))
{
}
  
void ScalarFieldBinaryOperator::execute(void) 
{
  //  cout << "ScalarFieldBinaryOperator::execute:start"<<endl;
  
  in_left = (FieldIPort *) get_iport("Scalar Field Left Operand");
  in_right = (FieldIPort *) get_iport("Scalar Field Right Operand");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle left_FH;
  FieldHandle right_FH;
  
  if(!in_left->get(left_FH)){
    error("Didn't get a handle to left field");
    return;
  } else if( !left_FH->query_scalar_interface(this).get_rep() ){
    error("Left input is not a Scalar field");
  }

  if(!in_right->get(right_FH)){
    error("Didn't get a handle to right field");
    return;
  } else if( !right_FH->query_scalar_interface(this).get_rep() ){
    error("Right input is not a Scalar field");
  }

  const SCIRun::TypeDescription *ftd = left_FH->get_type_description();
  CompileInfoHandle ci = 
     ScalarFieldBinaryOperatorAlgo::get_compile_info(ftd);
  Handle<ScalarFieldBinaryOperatorAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }

  FieldHandle fh =  algo->execute( left_FH, right_FH, guiOperation.get() );
      
  if( fh.get_rep() != 0 )
    sfout->send(fh);
  //  cout << "ScalarFieldBinaryOperator::execute:end\n";
}

CompileInfoHandle 
ScalarFieldBinaryOperatorAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarFieldBinaryOperatorAlgoT");
  static const string base_class_name("ScalarFieldBinaryOperatorAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  // Add in name space
  rval->add_namespace("Uintah");
  ftd->fill_compile_info(rval);
  return rval;
}


