
/*
 *  TensorsToIndices: Change a Field of indices (ints) into a Field or Tensors,
 *                      where the Tensor values are looked up in the
 *                      conductivity_table for each index
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Packages/BioPSE/Dataflow/Modules/Forward/TensorsToIndices.h>
#include <Core/Geometry/Tensor.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class TensorsToIndices : public Module {
  GuiInt outside_;
  GuiInt groundfirst_;
public:
  TensorsToIndices(GuiContext *context);
  virtual ~TensorsToIndices();
  virtual void execute();
};


DECLARE_MAKER(TensorsToIndices)


TensorsToIndices::TensorsToIndices(GuiContext *context)
  : Module("TensorsToIndices", context, Filter, "Forward", "BioPSE"),
    outside_(context->subVar("outside")),
    groundfirst_(context->subVar("groundfirst"))
{
}

TensorsToIndices::~TensorsToIndices()
{
}

void TensorsToIndices::execute() {
  FieldIPort* ifieldport = (FieldIPort *) get_iport("IndexField");
  FieldOPort* ofieldport = (FieldOPort *) get_oport("TensorField");
  if (!ifieldport) {
    error("Unable to initialize iport 'IndexField'.");
    return;
  }
  if (!ofieldport) {
    error("Unable to initialize iport 'TensorField'.");
    return;
  }

  FieldHandle ifieldH;
  if (!ifieldport->get(ifieldH))
    return;
  if (!ifieldH.get_rep()) {
    error("Empty input field.");
    return;
  }

  const TypeDescription *field_src_td = ifieldH->get_type_description();
  const string &field_src_name = field_src_td->get_name();
  string::size_type idx = field_src_name.find('<');
  string field_dst_name = field_src_name.substr(0,idx)+"<int>";
  cerr << "field_dst_name = "<<field_dst_name<<"\n";

  CompileInfoHandle ci =
    TensorsToIndicesAlgo::get_compile_info(field_src_td, field_dst_name);
  Handle<TensorsToIndicesAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldH = algo->execute(ifieldH);
  ofieldport->send(ofieldH);
}
} // End namespace BioPSE

namespace SCIRun {
CompileInfoHandle
TensorsToIndicesAlgo::get_compile_info(const TypeDescription *field_src_td,
				       const string &field_dst_name)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TensorsToIndicesAlgoT");
  static const string base_class_name("TensorsToIndicesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_src_td->get_filename() + "." +
		       to_filename(field_dst_name) + ".",
                       base_class_name, 
                       template_class_name, 
                       field_src_td->get_name() + "," + field_dst_name + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_src_td->fill_compile_info(rval);
  return rval;
}
} // End namespace SCIRun
