#include "TensorFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/StringUtil.h>

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
    guiEigen2DCalcType(ctx->subVar("eigen2D-calc-type")),
    guiNx(ctx->subVar("nx")),
    guiNy(ctx->subVar("ny")),
    guiNz(ctx->subVar("nz")),
    guiTx(ctx->subVar("tx")),
    guiTy(ctx->subVar("ty")),
    guiTz(ctx->subVar("tz"))
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

    scalarField = scinew LatVolField<double>(hTF->basis_order());

    performOperation( tensorField, scalarField );
    for(unsigned int i = 0; i < tensorField->nproperties(); i++){
      string prop_name(tensorField->get_property_name( i ));
      if(prop_name == "varname"){
        string prop_component;
        tensorField->get_property( prop_name, prop_component);
        switch(guiOperation.get()) {
        case 0: // extract element i,j
          scalarField->set_property("varname",
                                    string(prop_component + ":" +
                                           to_string( guiRow.get ()) + 
                                           "," + to_string( guiColumn.get ())),
                                    true);
          break;
        case 1: // extract eigen value
          scalarField->set_property("varname", 
                                    string(prop_component +":eigen"), true);
          break;
        case 2: // extract pressure
          scalarField->set_property("varname", 
                                    string(prop_component +":pressure"), true);
          break;
        case 3: // tensor stress
          scalarField->set_property("varname", 
                                    string(prop_component +":equiv_stress"), true);
          break;
        case 4: // tensor stress
          scalarField->set_property("varname",
                           string(prop_component +":sheer_stress"), true);
          break;
        case 5: // tensor stress
          scalarField->set_property("varname",
                           string(prop_component +"NdotSigmadotT"), true);
          break;
        default:
          scalarField->set_property("varname",
                                    string(prop_component.c_str()), true);
        }
      } else if( prop_name == "generation") {
        int generation;
        tensorField->get_property( prop_name, generation);
        scalarField->set_property(prop_name.c_str(), generation , true);
      } else if( prop_name == "timestep" ) {
        int timestep;
        tensorField->get_property( prop_name, timestep);
        scalarField->set_property(prop_name.c_str(), timestep , true);
      } else if( prop_name == "offset" ){
        IntVector offset(0,0,0);        
        tensorField->get_property( prop_name, offset);
        scalarField->set_property(prop_name.c_str(), IntVector(offset) , true);
      } else if( prop_name == "delta_t" ){
        double dt;
        tensorField->get_property( prop_name, dt);
        scalarField->set_property(prop_name.c_str(), dt , true);
      } else if( prop_name == "vartype" ){
        int vartype;
        tensorField->get_property( prop_name, vartype);
        scalarField->set_property(prop_name.c_str(), vartype , true);
      } else {
        warning( "Unknown field property, not transferred.");
      }
  }
    sfout->send(scalarField);
  }
}

} // end namespace Uintah



