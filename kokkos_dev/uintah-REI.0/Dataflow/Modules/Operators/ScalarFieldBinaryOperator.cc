#include "ScalarFieldBinaryOperator.h"

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include <math.h>

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


