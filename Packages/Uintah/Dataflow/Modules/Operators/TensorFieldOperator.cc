#include "TensorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Geometry/BBox.h>

//#include <SCICore/Math/Mat.h>


using namespace SCIRun;

namespace Uintah {
 
extern "C" Module* make_TensorFieldOperator( const string& id ) { 
  return scinew TensorFieldOperator( id );
}



TensorFieldOperator::TensorFieldOperator(const string& id)
  : Module("TensorFieldOperator",id,Source),
    guiOperation("operation", id, this),
    guiRow("row", id, this),
    guiColumn("column", id, this),
    guiPlaneSelect("planeSelect", id, this),
    guiDelta("delta", id, this),
    guiEigen2DCalcType("eigen2D-calc-type", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = scinew FieldIPort(this, "TensorField");
  sfout = scinew FieldOPort(this, "ScalarField");

  // Add ports to the Module
  add_iport(in);
  add_oport(sfout);
}
  
void TensorFieldOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "Matrix3" ){
    std::cerr<<"Input is not a Tensor field\n";
    return;
  }

  LatticeVol<double>  *scalarField = 0;  
  if( LevelField<Matrix3> *tensorField =
      dynamic_cast<LevelField<Matrix3>*>(hTF.get_rep())) {

    scalarField = scinew LatticeVol<double>(hTF->data_at());

    performOperation( tensorField, scalarField );
  }

  if( scalarField )
    sfout->send(scalarField);
}

} // end namespace Uintah



