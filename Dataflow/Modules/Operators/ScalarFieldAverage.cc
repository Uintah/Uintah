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
 
extern "C" Module* make_ScalarFieldAverage( const string& id ) { 
  return scinew ScalarFieldAverage( id );}


ScalarFieldAverage::ScalarFieldAverage(const string& id)
  : Module("ScalarFieldAverage",id,Source, "Operators", "Uintah"),
    t0_("t0_", id, this), t1_("t1_", id, this),
    tsteps_("tsteps_", id, this),
    aveField(0), varname(""), time(0)
{
}
  
void ScalarFieldAverage::execute(void) {
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
    std::cerr<<"Input is not a Scalar field.  type = "<<
      hTF->get_type_name(1)<<"\n";
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
  } else if( t != time ) {
    time = t;
    t1_.set( t );
    tsteps_.set( tsteps_.get() + 1 );
    reset_vars();
    averageField( hTF );
    // compute new average field
  } else {
    sfout->send(0);
    return;
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



