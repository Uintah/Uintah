#include "ScalarFieldAverage.h"
#include "ScalarOperatorFunctors.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Geometry/BBox.h>
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
	      hTF->get_type_name(1) != "long"){
    std::cerr<<"Input is not a Scalar field.  type = "<<
      hTF->get_type_name(1)<<"\n";
    return;
  }

  string vname;
  double t;
  
  if( !hTF->get( "variable", vname )){
    cerr<<"No variable in database"<<endl; }
  if ( !hTF->get( "time", t ) ){
    cerr<<"No time in database"<<endl; }

  if(aveField == 0) {
    aveField = new LatticeVol<double>(hTF->data_at());
    aveFieldH = aveField;
    varname = vname;
    cerr<<vname<<endl;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    fillField( hTF );
  } else if( vname != varname ){
    cerr<<vname<<endl;
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
  if( LevelField<double> *inField =
      dynamic_cast<LevelField<double>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  } else if( LevelField<float> *inField =
      dynamic_cast<LevelField<float>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  } else if( LevelField<long> *inField =
      dynamic_cast<LevelField<long>*>(fh.get_rep())) {
    initField( inField, aveField);
    computeScalars( inField, aveField, NoOp() ); 
    computeAverages( inField, aveField); 
  }
}

void ScalarFieldAverage::averageField( FieldHandle fh)
{
  if( LevelField<double> *inField =
      dynamic_cast<LevelField<double>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  } else if( LevelField<float> *inField =
      dynamic_cast<LevelField<float>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  } else if( LevelField<long> *inField =
      dynamic_cast<LevelField<long>*>(fh.get_rep())) {
    computeAverages( inField, aveField); 
  }
}

} // end namespace Uintah



