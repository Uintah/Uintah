#include "VectorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

//#include <SCICore/Math/Mat.h>


using namespace SCIRun;

namespace Uintah {
 
extern "C" Module* make_VectorFieldOperator( const string& id ) { 
  return scinew VectorFieldOperator( id );}


VectorFieldOperator::VectorFieldOperator(const string& id)
  : Module("VectorFieldOperator",id,Source, "Operators", "Uintah"),
    guiOperation("operation", id, this)
{
}
  
void VectorFieldOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Vector Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
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



