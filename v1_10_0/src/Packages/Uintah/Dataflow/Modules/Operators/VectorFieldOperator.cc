#include "VectorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

//#include <SCICore/Math/Mat.h>


using namespace SCIRun;

namespace Uintah {
 
  DECLARE_MAKER(VectorFieldOperator)

VectorFieldOperator::VectorFieldOperator(GuiContext* ctx)
  : Module("VectorFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}
  
void VectorFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Vector Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"VectorFieldOperator::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "Vector" ){
    std::cerr<<"Input is not a Vector field\n";
    return;
  }

  LatVolField<double>  *scalarField = 0;  
  if( LatVolField<Vector> *vectorField =
      dynamic_cast<LatVolField<Vector>*>(hTF.get_rep())) {

    scalarField = scinew LatVolField<double>(hTF->data_at());

    performOperation( vectorField, scalarField );
  }

  if( scalarField )
    sfout->send(scalarField);
}

} // end namespace Uintah



