#include "ScalarFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
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
	      hTF->get_type_name(1) != "long64"){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  FieldHandle fh = 0;
  if( LatticeVol<double> *scalarField1 =
      dynamic_cast<LatticeVol<double>*>(hTF.get_rep())) {
    LatticeVol<double>  *scalarField2 = 0;  

    scalarField2 = scinew LatticeVol<double>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatticeVol<float> *scalarField1 =
	     dynamic_cast<LatticeVol<float>*>(hTF.get_rep())) {
    LatticeVol<float>  *scalarField2 = 0;  

    scalarField2 = scinew LatticeVol<float>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatticeVol<long64> *scalarField1 =
	     dynamic_cast<LatticeVol<long64>*>(hTF.get_rep())) {
    LatticeVol<long64>  *scalarField2 = 0;  

    scalarField2 = scinew LatticeVol<long64>(hTF->data_at());
    performOperation( scalarField1, scalarField2 );
    fh = scalarField2;
  }
  if( fh.get_rep() != 0 )
    sfout->send(fh);
}

} // end namespace Uintah



