/*
 *  CastMLVtoHV.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Modules/Fields/CastMLVtoHV.h>
#include <Core/Containers/Handle.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE CastMLVtoHV : public Module {
private:
  int            last_gen_;
  FieldHandle    ofieldH_;

public:
  CastMLVtoHV(GuiContext* ctx);
  virtual ~CastMLVtoHV();
  virtual void execute();
};

  DECLARE_MAKER(CastMLVtoHV)

CastMLVtoHV::CastMLVtoHV(GuiContext* ctx)
  : Module("CastMLVtoHV", ctx, Filter, "Fields", "SCIRun"), last_gen_(-1)
{
}

CastMLVtoHV::~CastMLVtoHV(){
}



void CastMLVtoHV::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *iport_ = (FieldIPort*)get_iport("MaskedLatVolField");

  if (!iport_) {
    error("Unable to initialize iport 'MaskedLatVolField'.");
    return;
  }
  
  FieldOPort *oport_ = (FieldOPort*)get_oport("HexVolField");
  if (!oport_) {
    error("Unable to initialize oport 'HexVolField'.");
    return;
  }

  FieldHandle ifieldH;
  if (!iport_->get(ifieldH) || 
      !ifieldH.get_rep())
    return;
  
  if (ifieldH->generation == last_gen_) {
    oport_->send(ofieldH_);
    return;
  }
  last_gen_ = ifieldH->generation;

  // we expect that the input field is a MaskedLatVolField
  if (ifieldH.get_rep()->get_type_description(0)->get_name() !=
      "MaskedLatVolField")
  {
    error("Input volume is not a MaskedLatVolField.");
    return;
  }                     

  if (ifieldH->data_at() != Field::NODE) {
    error("Input volume data isn't node-centered.");
    return;
  }                         

  const TypeDescription *fsrc_td = ifieldH->get_type_description();
  const TypeDescription *lsrc_td;// = ifieldH->data_at_type_description();
  const TypeDescription *ldst_td = 0;
  switch (ifieldH->data_at())
  {
  case Field::NODE:
    ldst_td = get_type_description((HexVolMesh::Node *)0);
    break;

  case Field::EDGE:
    ldst_td = get_type_description((HexVolMesh::Edge *)0);
    break;

  case Field::FACE:
    ldst_td = get_type_description((HexVolMesh::Face *)0);
    break;

  case Field::CELL:
    ldst_td = get_type_description((HexVolMesh::Cell *)0);
    break;

  case Field::NONE:
    error("Unsupported data_at location.");
    return;
  }

  CompileInfoHandle ci =
    CastMLVtoHVAlgo::get_compile_info(fsrc_td, lsrc_td, ldst_td);
  Handle<CastMLVtoHVAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  ofieldH_ = algo->execute(ifieldH, ifieldH->data_at());

  oport_->send(ofieldH_);
}



CompileInfoHandle
CastMLVtoHVAlgo::get_compile_info(const TypeDescription *fsrc_td,
				  const TypeDescription *lsrc_td,
				  const TypeDescription *ldst_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CastMLVtoHVAlgoT");
  static const string base_class_name("CastMLVtoHVAlgo");

  const string::size_type loc = fsrc_td->get_name().find_first_of('<');
  const string fdst = "HexVolField" + fsrc_td->get_name().substr(loc);

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc_td->get_filename() + "." +
		       lsrc_td->get_filename() + "." +
		       to_filename(fdst) + "." +
		       ldst_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc_td->get_name() + ", " +
		       lsrc_td->get_name() + ", " +
                       fdst + ", " +
		       ldst_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


