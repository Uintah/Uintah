#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/Schlieren.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include <math.h>

using namespace SCIRun;

namespace Uintah {
class Schlieren: public Module
{
public:
  Schlieren(GuiContext* ctx);
  virtual ~Schlieren() {}
    
  virtual void execute(void);
    
private:

  GuiDouble dx_, dy_, dz_;

  FieldIPort * in_;
  FieldOPort * sfout_;
};
} //end namespace Uintah

using namespace Uintah;

DECLARE_MAKER(Schlieren)

Schlieren::Schlieren(GuiContext* ctx)
  : Module("Schlieren",ctx,Source, "Operators", "Uintah"),
    dx_(get_ctx()->subVar("dx")), dy_(get_ctx()->subVar("dy")), dz_(get_ctx()->subVar("dz"))
{
}
  
void
Schlieren::execute(void) 
{
  in_    = (FieldIPort *) get_iport("Scalar Field");
  sfout_ = (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  // bullet proofing
  if(!in_->get(hTF)){
    std::cerr<<"Schlieren::execute(void) Didn't get a handle\n";
    return;
  } else if ( !hTF->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field");
    return;
  } else if ( hTF->basis_order() != 0 ) {
    error("Input must be cell centered (basis order = 0 ).");
    return;
  }

  // WARNING: will not yet work on a Mult-level Dataset!!!!

  //##################################################################


  const SCIRun::TypeDescription *sftd = hTF->get_type_description();
  CompileInfoHandle ci = SchlierenAlgo::get_compile_info(sftd);
  Handle<SchlierenAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }

  //##################################################################    

  FieldHandle fh =  algo->execute( hTF, dx_.get(), dy_.get(), dz_.get() );
  if( fh.get_rep() != 0 ){
    sfout_->send(fh);
  }
}

CompileInfoHandle
SchlierenAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SchlierenAlgoT");
  static const string base_class_name("SchlierenAlgo");
  
  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}
