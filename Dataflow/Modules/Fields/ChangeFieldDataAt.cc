/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


//    File   : ChangeFieldDataAt.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataAt.h>
#include <Dataflow/Modules/Fields/ApplyInterpMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {


class PSECORESHARE ChangeFieldDataAt : public Module {
public:
  GuiString outputdataat_;    // the out data at
  GuiString inputdataat_;     // the in data at
  GuiString fldname_;         // the field name
  int              generation_;
  ChangeFieldDataAt(GuiContext* ctx);
  virtual ~ChangeFieldDataAt();
  virtual void execute();
  void update_input_attributes(FieldHandle);
};

  DECLARE_MAKER(ChangeFieldDataAt)

ChangeFieldDataAt::ChangeFieldDataAt(GuiContext* ctx)
  : Module("ChangeFieldDataAt", ctx, Filter, "FieldsData", "SCIRun"),
    outputdataat_(ctx->subVar("outputdataat")),
    inputdataat_(ctx->subVar("inputdataat", false)),
    fldname_(ctx->subVar("fldname", false)),
    generation_(-1)
{
}

ChangeFieldDataAt::~ChangeFieldDataAt()
{
  fldname_.set("---");
  inputdataat_.set("---");
}


void
ChangeFieldDataAt::update_input_attributes(FieldHandle f) 
{
  switch(f->basis_order())
  {
  case 0: 
    inputdataat_.set("Cells");
    break;
  case 1:
    inputdataat_.set("Nodes");
    break;
  default: ;
  }

  string fldname;
  if (f->get_property("name",fldname))
  {
    fldname_.set(fldname);
  }
  else
  {
    fldname_.set("--- No Name ---");
  }
}


void
ChangeFieldDataAt::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 
  if (!iport) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  
  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    fldname_.set("---");
    inputdataat_.set("---");
    return;
  }

  if (generation_ != fh.get_rep()->generation)
  {
    update_input_attributes(fh);
    generation_ = fh.get_rep()->generation;
  }

  // The output port is required.
  FieldOPort *ofport = (FieldOPort*)get_oport("Output Field");
  if (!ofport) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  // The output port is required.
  MatrixOPort *omport = (MatrixOPort*)get_oport("Interpolant");
  if (!omport) {
    error("Unable to initialize oport 'Interpolant'.");
    return;
  }

  int basis_order = fh->basis_order();
  const string &d = outputdataat_.get();
  if (d == "Nodes")
  {
    basis_order = 1;
  }
  else if (d == "Cells")
  {
    basis_order = 0;
  }
  if (basis_order == fh->basis_order())
  {
    // No changes, just send the original through (it may be nothing!).
    remark("Passing field from input port to output port unchanged.");
    warning("Interpolant for that location combination is not yet supported.");
    ofport->send(fh);
    omport->send(0);
    return;
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrctd = fh->get_type_description();
  CompileInfoHandle ci =
    ChangeFieldDataAtAlgoCreate::get_compile_info(fsrctd);
  Handle<ChangeFieldDataAtAlgoCreate> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  update_state(Executing);
  MatrixHandle interpolant(0);
  FieldHandle ef(algo->execute(this, fh, basis_order, interpolant));

  // Automatically apply the interpolant matrix to the output field.
  if (ef.get_rep() && interpolant.get_rep())
  {
    string actype = fh->get_type_description(1)->get_name();
    if (fh->query_scalar_interface(this) != NULL) { actype = "double"; }
    const TypeDescription *iftd = fh->get_type_description();
    const TypeDescription *iltd = fh->order_type_description();
    const TypeDescription *oftd = ef->get_type_description();
    const TypeDescription *oltd = ef->order_type_description();
    CompileInfoHandle ci =
      ApplyInterpMatrixAlgo::get_compile_info(iftd, iltd,
					      oftd, oltd,
					      actype, false);
    Handle<ApplyInterpMatrixAlgo> algo;
    if (module_dynamic_compile(ci, algo))
    {
      algo->execute_aux(fh, ef, interpolant);
    }
  }

  if (interpolant.get_rep() == 0)
  {
    if (omport->nconnections() > 0)
    {
      error("Interpolant for that location combination is not supported.");
    }
    else
    {
      warning("Interpolant for that location combination is not supported.");
    }
  }

  ofport->send(ef);
  omport->send(interpolant);
}

    

CompileInfoHandle
ChangeFieldDataAtAlgoCreate::get_compile_info(const TypeDescription *field_td)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldDataAtAlgoCreateT");
  static const string base_class_name("ChangeFieldDataAtAlgoCreate");

  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       field_td->get_filename() + ".",
		       base_class_name, 
		       template_class,
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace Moulding


