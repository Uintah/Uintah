#include "ScalarFieldOperator.h"
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
 
extern "C" Module* make_ScalarFieldOperator( const string& id ) { 
  return scinew ScalarFieldOperator( id );}


ScalarFieldOperator::ScalarFieldOperator(const string& id)
  : Module("ScalarFieldOperator",id,Source, "Operators", "Uintah"),
    guiOperation("operation", id, this)
{
}
  
void ScalarFieldOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Scalar Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "double" &&
	      hTF->get_type_name(1) != "float" &&
	      hTF->get_type_name(1) != "long"){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  LatticeVol<double>  *scalarField2 = 0;  
  if( LevelField<double> *scalarField1 =
      dynamic_cast<LevelField<double>*>(hTF.get_rep())) {

    scalarField2 = scinew LatticeVol<double>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );

  } else if( LevelField<float> *scalarField1 =
      dynamic_cast<LevelField<float>*>(hTF.get_rep())) {

    scalarField2 = scinew LatticeVol<double>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );

  } else if( LevelField<long> *scalarField1 =
      dynamic_cast<LevelField<long>*>(hTF.get_rep())) {

    scalarField2 = scinew LatticeVol<double>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
  }
  if( scalarField2 )
    sfout->send(scalarField2);
}

} // end namespace Uintah



