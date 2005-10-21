#include "ScalarFieldOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>



using namespace SCIRun;

namespace Uintah {
 
class ScalarFieldOperator: public Module
{
public:
  
  ScalarFieldOperator(GuiContext* ctx);
  virtual ~ScalarFieldOperator() {}
    
  virtual void execute(void);
    
private:
  //    TCLstring tcl_status;
  GuiInt guiOperation;

  FieldIPort *in;

  FieldOPort *sfout;
  //ScalarFieldOPort *vfout;
};

} // end namespace Uintah

using namespace Uintah;

DECLARE_MAKER(ScalarFieldOperator)

ScalarFieldOperator::ScalarFieldOperator(GuiContext* ctx)
  : Module("ScalarFieldOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}

template<class T1, class T2> 
void
ScalarFieldOperatorAlgo::set_properties( T1* sf1, T2* sf2)
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
  if(!in->get(hTF) ){
    error("execute(void) Didn't get a handle.");
    return;
  } else if ( !hTF->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field.");
    return;
  }

  
  const SCIRun::TypeDescription *ftd = hTF->get_type_description();
  CompileInfoHandle ci = ScalarFieldOperatorAlgo::get_compile_info(ftd);
  Handle<ScalarFieldOperatorAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }



  FieldHandle fh =  algo->execute( hTF, guiOperation );
  if( fh.get_rep() != 0 ){
    sfout->send(fh);
  }
}


CompileInfoHandle
ScalarFieldOperatorAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarFieldOperatorAlgoT");
  static const string base_class_name("ScalarFieldOperatorAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}




