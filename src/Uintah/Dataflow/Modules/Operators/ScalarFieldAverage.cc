/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>

#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include "ScalarFieldAverage.h"

#include <cmath>
#include <iostream>

using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Uintah {
class ScalarFieldAverage: public Module {
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type                 LVMeshHandle;
  typedef HexTrilinearLgn<double>             FDdoubleBasis;
  typedef ConstantBasis<double>               CFDdoubleBasis; 

  typedef GenericField<LVMesh, CFDdoubleBasis, FData3d<double, LVMesh> > CDField;
  typedef GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> > LDField;

  ScalarFieldAverage(GuiContext* ctx);
  virtual ~ScalarFieldAverage() {}
    
  virtual void execute(void);
private:
  
  GuiDouble t0_;
  GuiDouble t1_;
  GuiInt tsteps_;
  
  FieldIPort *in;
  FieldOPort *sfout;
  
  FieldHandle  aveFieldHandle;
  string varname;
  double time;
  //ScalarFieldOPort *vfout;
   

};
} // end namespace Uintah

using namespace Uintah;
DECLARE_MAKER(ScalarFieldAverage)

ScalarFieldAverage::ScalarFieldAverage(GuiContext* ctx)
  : Module("ScalarFieldAverage",ctx,Source, "Operators", "Uintah"),
    t0_(get_ctx()->subVar("t0_")), t1_(get_ctx()->subVar("t1_")),
    tsteps_(get_ctx()->subVar("tsteps_")),
    aveFieldHandle(0), varname(""), time(0)
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
  } else if ( !hTF->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field.");
    return;
  }

  string vname;
  double t;
  
  if( !hTF->get_property( "variable", vname )){
    cerr<<"No variable in database"<<endl; }
  if ( !hTF->get_property( "time", t ) ){
    cerr<<"No time in database"<<endl; }

  const SCIRun::TypeDescription *td1 = hTF->get_type_description();
  if( td1->get_name().find("LatVolMesh") == string::npos ){
    error("Field is not a LatVolMesh based field");
    return;
  }
  // should be safe because of the above if
  LVMeshHandle mh((LVMesh*)hTF->mesh().get_rep());


  CompileInfoHandle ci;
  Handle<ScalarFieldAverageAlgo> algo;

  const SCIRun::TypeDescription *td2;
  if( hTF->basis_order() == 0 && aveFieldHandle == 0){
    mh.detach();
    aveFieldHandle = scinew CDField( mh );
    td2 = aveFieldHandle->get_type_description();
    ci = ScalarFieldAverageAlgo::get_compile_info(td1, td2);
    if( !module_dynamic_compile(ci, algo) ){
      error("dynamic compile failed.");
      return;
    }
    varname = vname;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    algo->fillField( hTF, aveFieldHandle );
  } else if(hTF->basis_order() == 1 && aveFieldHandle == 0){
    mh.detach();
    aveFieldHandle = scinew LDField( mh );
    td2 = aveFieldHandle->get_type_description();
    ci = ScalarFieldAverageAlgo::get_compile_info(td1, td2);
    if( !module_dynamic_compile(ci, algo) ){
      error("dynamic compile failed.");
      return;
    }
    varname = vname;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    algo->fillField( hTF, aveFieldHandle );
  } else if( vname != varname ){
    td2 = aveFieldHandle->get_type_description();
    ci = ScalarFieldAverageAlgo::get_compile_info(td1, td2);
    if( !module_dynamic_compile(ci, algo) ){
      error("dynamic compile failed.");
      return;
    }
    varname = vname;
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    algo->fillField( hTF, aveFieldHandle );
  } else if( t < time){
    td2 = aveFieldHandle->get_type_description();
    ci = ScalarFieldAverageAlgo::get_compile_info(td1, td2);
    if( !module_dynamic_compile(ci, algo) ){
      error("dynamic compile failed.");
      return;
    }
    time = t;
    t0_.set( t );
    t1_.set( t );
    tsteps_.set( 1 );
    reset_vars();
    algo->fillField( hTF, aveFieldHandle );    
  } else if( t > time ) {
    td2 = aveFieldHandle->get_type_description();
    ci = ScalarFieldAverageAlgo::get_compile_info(td1, td2);
    if( !module_dynamic_compile(ci, algo) ){
      error("dynamic compile failed.");
      return;
    }
    time = t;
    t1_.set( t );
    tsteps_.set( tsteps_.get() + 1 );
    reset_vars();
    algo->averageField( hTF, aveFieldHandle );
    // compute new average field
  }
  sfout->send(aveFieldHandle);
}


CompileInfoHandle
ScalarFieldAverageAlgo::get_compile_info(const SCIRun::TypeDescription *td1,
                                         const SCIRun::TypeDescription *td2)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarFieldAverageAlgoT");
  static const string base_class_name("ScalarFieldAverageAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       td1->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       td1->get_name() + "," +
                       td2->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  // add the namespace
  rval->add_namespace("Uintah");
  td1->fill_compile_info(rval);
  td2->fill_compile_info(rval);
  return rval;
}


