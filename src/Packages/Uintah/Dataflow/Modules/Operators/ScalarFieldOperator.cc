#include "ScalarFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
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
	      hTF->get_type_name(1) != "long64"){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  FieldHandle fh = 0;
  if( LatVolField<double> *scalarField1 =
      dynamic_cast<LatVolField<double>*>(hTF.get_rep())) {
    LatVolField<double>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<double>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatVolField<float> *scalarField1 =
	     dynamic_cast<LatVolField<float>*>(hTF.get_rep())) {
    LatVolField<float>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<float>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatVolField<long64> *scalarField1 =
	     dynamic_cast<LatVolField<long64>*>(hTF.get_rep())) {
    LatVolField<long64>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<long64>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  }
  if( fh.get_rep() != 0 )
    sfout->send(fh);
}

} // end namespace Uintah



