/*
 *  Silhoettes.cc:
 *
 *  Written by:
 *   allen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Handle.h>

#include <Packages/PCS/Dataflow/Modules/Visualization/Silhoettes.h>

namespace PCS {

using namespace SCIRun;

class PSECORESHARE Silhoettes : public Module {
public:
  Silhoettes(GuiContext*);

  virtual ~Silhoettes();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

protected:
  FieldHandle fieldout_;

  int fGeneration_;
};


DECLARE_MAKER(Silhoettes)
Silhoettes::Silhoettes(GuiContext* ctx)
  : Module("Silhoettes", ctx, Source, "Visualization", "PCS"),
    fGeneration_(-1)
{
}

Silhoettes::~Silhoettes(){
}

void
Silhoettes::execute(){
  FieldIPort* ifp = (FieldIPort *)get_iport("Input Field");

  FieldHandle fieldin;

  if (!ifp)
  {
    error( "Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifp->get(fieldin) && fieldin.get_rep()))
  {
    error( "No handle or representation in input field." );
    return;
  }

  TypeDescription *otd = 0;

  if (!fieldin->query_scalar_interface(this).get_rep() ) {
    error( "This module only works on fields of scalar data.");
    return;
  }

  // If no data or a changed recalcute.
  if( !fieldout_.get_rep() ||
      fGeneration_ != fieldin->generation ) {
    fGeneration_ = fieldin->generation;

    const TypeDescription *ftd = fieldin->get_type_description(0);
    const TypeDescription *ttd = fieldin->get_type_description(1);

    CompileInfoHandle ci =
      SilhoettesAlgo::get_compile_info(ftd, ttd);
    Handle<SilhoettesAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fieldout_ = algo->execute(fieldin);
  }

  // Get a handle to the output field port.
  if ( fieldout_.get_rep() ) {
    FieldOPort* ofp = (FieldOPort *) get_oport("Silhoettes");

    if (!ofp) {
      error("Unable to initialize oport 'Silhoettes'.");
      return;
    }

    // Send the data downstream
    ofp->send(fieldout_);
  }
}

void
Silhoettes::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

CompileInfoHandle
SilhoettesAlgo::get_compile_info(const TypeDescription *ftd,
				 const TypeDescription *ttd )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SilhoettesAlgoT");
  static const string base_class_name("SilhoettesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + "<" + ttd->get_name() + "> " + ", " +
                       "CurveField" + "<" + ttd->get_name() + "> " + ", " +
                       "CurveMesh" );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("SCIRun");
  rval->add_namespace("PCS");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace PCS


