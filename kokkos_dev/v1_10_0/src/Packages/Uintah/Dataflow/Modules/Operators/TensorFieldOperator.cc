#include "TensorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

//#include <SCICore/Math/Mat.h>
#include <iostream>
using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Uintah {
 
  DECLARE_MAKER(TensorFieldOperator)


TensorFieldOperator::TensorFieldOperator(GuiContext* ctx)
  : Module("TensorFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation")),
    guiRow(ctx->subVar("row")),
    guiColumn(ctx->subVar("column")),
    guiPlaneSelect(ctx->subVar("planeSelect")),
    guiDelta(ctx->subVar("delta")),
    guiEigen2DCalcType(ctx->subVar("eigen2D-calc-type"))
    //    tcl_status(ctx->subVar("tcl_status")),
{
}
  
void TensorFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 

  in = (FieldIPort *) get_iport("Tensor Field");
  sfout = (FieldOPort *) get_oport("Scalar Field");
  
  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"TensorFieldOperator::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "Matrix3" ){
    std::cerr<<"Input is not a Tensor field\n";
    return;
  }


    
  LatVolField<double>  *scalarField = 0;  
  if( LatVolField<Matrix3> *tensorField =
      dynamic_cast<LatVolField<Matrix3>*>(hTF.get_rep())) {

    scalarField = scinew LatVolField<double>(hTF->data_at());

    performOperation( tensorField, scalarField );
    sfout->send(scalarField);
  }
}

} // end namespace Uintah



