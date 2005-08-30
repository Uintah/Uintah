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
 
DECLARE_MAKER(ScalarFieldOperator)

ScalarFieldOperator::ScalarFieldOperator(GuiContext* ctx)
  : Module("ScalarFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}

template<class T1, class T2> 
void
ScalarFieldOperator::set_properties( T1* sf1, T2* sf2)
{
  for(size_t i = 0; i < sf1->nproperties(); i++){
    string prop_name(sf1->get_property_name( i ));
    if(prop_name == "varname"){
      string prop_component;
      sf1->get_property( prop_name, prop_component);
      switch(guiOperation.get()) {
      case 0: // extract element 1
        sf2->set_property("varname",
                          string(prop_component +":ln"), true);
        break;
      case 1: // extract element 2
        sf2->set_property("varname", 
                          string(prop_component +":e"), true);
        break;
      default:
        sf2->set_property("varname",
                          string(prop_component.c_str()), true);
      }
    } else if( prop_name == "generation") {
      int generation;
      sf1->get_property( prop_name, generation);
      sf2->set_property(prop_name.c_str(), generation , true);
    } else if( prop_name == "timestep" ) {
      int timestep;
      sf1->get_property( prop_name, timestep);
      sf2->set_property(prop_name.c_str(), timestep , true);
    } else if( prop_name == "offset" ){
      IntVector offset(0,0,0);        
      sf1->get_property( prop_name, offset);
      sf2->set_property(prop_name.c_str(), IntVector(offset) , true);
    } else if( prop_name == "delta_t" ){
      double dt;
      sf1->get_property( prop_name, dt);
      sf2->set_property(prop_name.c_str(), dt , true);
    } else if( prop_name == "vartype" ){
      int vartype;
      sf1->get_property( prop_name, vartype);
      sf2->set_property(prop_name.c_str(), vartype , true);
    } else {
      warning( "Unknown field property, not transferred.");
    }
  }
}
  
void ScalarFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Scalar Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"ScalarFieldOperator::execute(void) Didn't get a handle\n";
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

    scalarField2 = scinew LatVolField<double>(hTF->basis_order());
    performOperation( scalarField1, scalarField2 );
    set_properties(scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatVolField<float> *scalarField1 =
	     dynamic_cast<LatVolField<float>*>(hTF.get_rep())) {
    LatVolField<float>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<float>(hTF->basis_order());
    performOperation( scalarField1, scalarField2 );
    set_properties(scalarField1, scalarField2 );
    fh = scalarField2;
  } else if( LatVolField<long64> *scalarField1 =
	     dynamic_cast<LatVolField<long64>*>(hTF.get_rep())) {
    LatVolField<long64>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<long64>(hTF->basis_order());
    performOperation( scalarField1, scalarField2 );
    set_properties(scalarField1, scalarField2 );
    fh = scalarField2;
  }
  if( fh.get_rep() != 0 ){

  }
}


} // end namespace Uintah



