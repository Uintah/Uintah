/*
 *  Centroids.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Fields/Centroids.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Util/DynamicCompilation.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE Centroids : public Module {
public:
  Centroids(GuiContext* ctx);
  virtual ~Centroids();
  virtual void execute();
};


DECLARE_MAKER(Centroids)


Centroids::Centroids(GuiContext* ctx)
  : Module("Centroids", ctx, Filter, "FieldsCreate", "SCIRun")
{
}


Centroids::~Centroids()
{
}



void
Centroids::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("TetVolField");

  if (!ifieldPort) {
    error("Unable to initialize iport 'TetVolField'.");
    return;
  }
  FieldHandle ifieldhandle;
  if (!ifieldPort->get(ifieldhandle) || !ifieldhandle.get_rep()) return;

  FieldOPort *ofieldPort = (FieldOPort*)get_oport("PointCloudField");
  if (!ofieldPort) {
    error("Unable to initialize oport 'PointCloudField'.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = CentroidsAlgo::get_compile_info(ftd);
  Handle<CentroidsAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle));
  
  ofieldPort->send(ofieldhandle);
}



CompileInfoHandle
CentroidsAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CentroidsAlgoT");
  static const string base_class_name("CentroidsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun


