#include "ScalarFieldAverage.h"
#include "ScalarOperatorFunctors.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <iostream>
//#include <SCICore/Math/Mat.h>
using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Uintah {
 
DECLARE_MAKER(ScalarFieldAverage)

ScalarFieldAverage::ScalarFieldAverage(GuiContext* ctx)
  : Module("ScalarFieldAverage",ctx,Source, "Operators", "Uintah"),
    t0_(ctx->subVar("t0_")), t1_(ctx->subVar("t1_")),
    tsteps_(ctx->subVar("tsteps_")),
    aveField(0), varname(""), time(0)
{
}
  
void ScalarFieldAverage::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (FieldIPort *) get_iport("Scalar Field");

  sfout =  (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    error("ScalarFieldAverage::execute(void) Didn't get a handle.");
    return;
  } else if ( hTF->get_type_name(1) != "double" &&
	      hTF->get_type_name(1) != "float" &&
	      hTF->get_type_name(1) != "long64"){
    error("Input is not an accetpable field type.");
      //      hTF->get_type_name(1)<<"\n";
    return;
  }

  string vname;
  double t;
  
  if( !hTF->get_property( "variable", vname )){
    cerr<<"No variable in database"<<endl; }
  if ( !hTF->get_property( "time", t ) ){
    cerr<<"No time in database"<<endl; }

  if(aveField == 0) {
    aveField = new LatVolField<double>(hTF->data_at());
    aveFieldH = aveField;
    varname = vname;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    fillField( hTF );
  } else if( vname != varname ){
    varname = vname;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    fillField( hTF );
  } else if( t < time ){
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    fillField( hTF );    
  } else if( t > time ) {
    time = t;
    t1_.set( t );
    tsteps_.set( tsteps_.get() + 1 );
    reset_vars();
    averageField( hTF );
    // compute new average field
  }
  sfout->send(aveField);
}

void ScalarFieldAverage::fillField( FieldHandle fh )
{
  if( LatVolField<double> *inField =
      dynamic_cast<LatVolField<double>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  } else if( LatVolField<float> *inField =
      dynamic_cast<LatVolField<float>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  } else if( LatVolField<long64> *inField =
      dynamic_cast<LatVolField<long64>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  }
}

void ScalarFieldAverage::averageField( FieldHandle fh)
{
  if( LatVolField<double> *inField =
      dynamic_cast<LatVolField<double>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  } else if( LatVolField<float> *inField =
      dynamic_cast<LatVolField<float>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  } else if( LatVolField<long64> *inField =
      dynamic_cast<LatVolField<long64>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  }
}

} // end namespace Uintah



