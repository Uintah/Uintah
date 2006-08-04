#include "TensorFieldOperator.h"

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <math.h>
#include <iostream>
using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Uintah {
 class TensorFieldOperator: public Module
{
public:

  TensorFieldOperator(GuiContext* ctx);
  virtual ~TensorFieldOperator() {}
    
  virtual void execute(void);
    
private:

  //    TCLstring tcl_status;
  GuiInt guiOperation;

  // element extractor operation
  GuiInt guiRow;
  GuiInt guiColumn;
    
    // eigen value/vector operation
    //GuiInt guiEigenSelect;

    // eigen 2D operation
  GuiInt guiPlaneSelect;
  GuiDouble guiDelta;
  GuiInt guiEigen2DCalcType;
    
  // n . sigma . t operation
  GuiDouble guiNx, guiNy, guiNz;
  GuiDouble guiTx, guiTy, guiTz;

  FieldIPort *in;

  FieldOPort *sfout;
  //VectorFieldOPort *vfout;
    
};
} // end namespace Uintah

using namespace Uintah;

DECLARE_MAKER(TensorFieldOperator)


TensorFieldOperator::TensorFieldOperator(GuiContext* ctx)
  : Module("TensorFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(get_ctx()->subVar("operation")),
    guiRow(get_ctx()->subVar("row")),
    guiColumn(get_ctx()->subVar("column")),
    guiPlaneSelect(get_ctx()->subVar("planeSelect")),
    guiDelta(get_ctx()->subVar("delta")),
    guiEigen2DCalcType(get_ctx()->subVar("eigen2D-calc-type")),
    guiNx(get_ctx()->subVar("nx")),
    guiNy(get_ctx()->subVar("ny")),
    guiNz(get_ctx()->subVar("nz")),
    guiTx(get_ctx()->subVar("tx")),
    guiTy(get_ctx()->subVar("ty")),
    guiTz(get_ctx()->subVar("tz"))
    //    tcl_status(get_ctx()->subVar("tcl_status")),
{
}
  
void TensorFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 

  in = (FieldIPort *) get_iport("Tensor Field");
  sfout = (FieldOPort *) get_oport("Scalar Field");
  
  FieldHandle hTF;
  
  if(!in->get(hTF)){
    error("TensorFieldOperator::execute(void) Didn't get a handle");
    return;
  }

  const SCIRun::TypeDescription *tftd =  hTF->get_type_description();
  if ( tftd->get_name().find("Matrix3") == string::npos ){
    error("Input is not a Tensor field");
    return;
  }

  //##################################################################


  CompileInfoHandle ci = TensorFieldOperatorAlgo::get_compile_info(tftd);
  Handle<TensorFieldOperatorAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }

  //##################################################################    

  algo->set_values(guiRow.get(), guiColumn.get(), guiPlaneSelect.get(),
                   guiDelta.get(), guiEigen2DCalcType.get(),
                   guiNx.get(), guiNy.get(), guiNz.get(),
                   guiTx.get(), guiTy.get(), guiTz.get());
  FieldHandle fh =  algo->execute( hTF, guiOperation.get() );
  if( fh.get_rep() != 0 ){
    sfout->send(fh);
  }


}

CompileInfoHandle
TensorFieldOperatorAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TensorFieldOperatorAlgoT");
  static const string base_class_name("TensorFieldOperatorAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  // New compilers need include files ordered.  Sometimes we
  // have to add them here so they are included in the right order.
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  // add namespace
  rval->add_namespace("Uintah");
  ftd->fill_compile_info(rval);
  return rval;
}




