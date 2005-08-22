#include "VectorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

#include <iostream>

//#include <SCICore/Math/Mat.h>


using namespace SCIRun;
using std::cerr;

namespace Uintah {
 
  DECLARE_MAKER(VectorFieldOperator)

VectorFieldOperator::VectorFieldOperator(GuiContext* ctx)
  : Module("VectorFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}
  
void
VectorFieldOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Vector Field");
  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    error( "VectorFieldOperator::execute(): Didn't get a handle!\n" );
    std::cerr<<"VectorFieldOperator::execute(void) Didn't get a handle\n";
    return;
  } else if ( !hTF.get_rep() ){
    warning( "VectorFieldOperator::execute(): Input is empty!\n" );
    return;
  } else if ( hTF->get_type_name(1) != "Vector" ){
    error( "VectorFieldOperator::execute(): Input is not a Vector Field!\n" );
    std::cerr<<"Input is not a Vector field\n";
    return;
  }

  // WARNING: will not yet work on a Mult-level Dataset!!!!

  LatVolField<double>  *scalarField = 0;  
  if( LatVolField<Vector> *vectorField =
      dynamic_cast<LatVolField<Vector>*>(hTF.get_rep())) {

    scalarField = scinew LatVolField<double>(hTF->basis_order());

    performOperation( vectorField, scalarField );
    
    for(unsigned int i = 0; i < vectorField->nproperties(); i++){
      string prop_name(vectorField->get_property_name( i ));
      if(prop_name == "varname"){
        string prop_component;
        vectorField->get_property( prop_name, prop_component);
        switch(guiOperation.get()) {
        case 0: // extract element 1
          scalarField->set_property("varname",
                                    string(prop_component +":1"), true);
          break;
        case 1: // extract element 2
          scalarField->set_property("varname", 
                                    string(prop_component +":2"), true);
          break;
        case 2: // extract element 3
          scalarField->set_property("varname", 
                                    string(prop_component +":3"), true);
          break;
        case 3: // Vector length
          scalarField->set_property("varname", 
                                    string(prop_component +":length"), true);
          break;
        case 4: // Vector curvature
          scalarField->set_property("varname",
                           string(prop_component +":vorticity"), true);
          break;
        default:
          scalarField->set_property("varname",
                                    string(prop_component.c_str()), true);
        }
      } else if( prop_name == "generation") {
        int generation;
        vectorField->get_property( prop_name, generation);
        scalarField->set_property(prop_name.c_str(), generation , true);
      } else if( prop_name == "timestep" ) {
        int timestep;
        vectorField->get_property( prop_name, timestep);
        scalarField->set_property(prop_name.c_str(), timestep , true);
      } else if( prop_name == "offset" ){
        IntVector offset(0,0,0);        
        vectorField->get_property( prop_name, offset);
        scalarField->set_property(prop_name.c_str(), IntVector(offset) , true);
        cerr<<"vector offset is "<< offset <<"n";
      } else if( prop_name == "delta_t" ){
        double dt;
        vectorField->get_property( prop_name, dt);
        scalarField->set_property(prop_name.c_str(), dt , true);
      } else if( prop_name == "vartype" ){
        int vartype;
        vectorField->get_property( prop_name, vartype);
        scalarField->set_property(prop_name.c_str(), vartype , true);
      } else {
        warning( "Unknown field property, not transferred.");
      }
    }
  }   

  if( scalarField )
    sfout->send(scalarField);
}

} // end namespace Uintah



