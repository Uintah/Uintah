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


//    File   : ChangeFieldBasis.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Modules/Fields/ChangeFieldBasis.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {


class ChangeFieldBasis : public Module {
public:
  GuiString outputdataat_;    // the out data at
  GuiString inputdataat_;     // the in data at
  GuiString fldname_;         // the field name
  int              generation_;
  ChangeFieldBasis(GuiContext* ctx);
  virtual ~ChangeFieldBasis();
  virtual void execute();
  void update_input_attributes(FieldHandle);
};

  DECLARE_MAKER(ChangeFieldBasis)

ChangeFieldBasis::ChangeFieldBasis(GuiContext* ctx)
  : Module("ChangeFieldBasis", ctx, Filter, "FieldsData", "SCIRun"),
    outputdataat_(ctx->subVar("output-basis")),
    inputdataat_(ctx->subVar("inputdataat", false)),
    fldname_(ctx->subVar("fldname", false)),
    generation_(-1)
{
}

ChangeFieldBasis::~ChangeFieldBasis()
{
  fldname_.set("---");
  inputdataat_.set("---");
}


void
ChangeFieldBasis::update_input_attributes(FieldHandle f) 
{
  static char *at_table[4] = { "Nodes", "Edges", "Faces", "Cells" };
  switch(f->basis_order())
  {
  case 1:
    inputdataat_.set("Nodes (linear basis)");
    break;
  case 0:
    inputdataat_.set(at_table[f->mesh()->dimensionality()] +
                     string(" (constant basis)"));
    break;
  case -1: 
    inputdataat_.set("None");
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
ChangeFieldBasis::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input"); 
  
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
  FieldOPort *ofport = (FieldOPort*)get_oport("Output");

  // The output port is required.
  MatrixOPort *omport = (MatrixOPort*)get_oport("Mapping");

  int basis_order = fh->basis_order();
  const string &bstr = outputdataat_.get();
  if (bstr == "Linear")
  {
    if (fh->mesh()->dimensionality() == 0)
    {
      basis_order = 0;
    }
    else
    {
      basis_order = 1;
    }
  }
  else if (bstr == "Constant")
  {
    basis_order = 0;
  }
  else if (bstr == "None")
  {
    basis_order = -1;
  }

  if (basis_order == fh->basis_order())
  {
    // No changes, just send the original through (it may be nothing!).
    remark("Passing field from input port to output port unchanged.");
    ofport->send(fh);

    MatrixHandle m(SparseRowMatrix::identity(fh->data_size()));
    omport->send(m);

    return;
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrctd = fh->get_type_description();
  CompileInfoHandle ci =
    ChangeFieldBasisAlgoCreate::get_compile_info(fsrctd);
  Handle<ChangeFieldBasisAlgoCreate> algo;
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
      ApplyMappingMatrixAlgo::get_compile_info(iftd, iltd,
					      oftd, oltd,
					      actype, false);
    Handle<ApplyMappingMatrixAlgo> algo;
    if (module_dynamic_compile(ci, algo))
    {
      algo->execute_aux(fh, ef, interpolant);
    }
  }

  if (interpolant.get_rep() == 0)
  {
    if (omport->nconnections() > 0)
    {
      error("Mapping for that location combination is not supported.");
    }
    else
    {
      warning("Mapping for that location combination is not supported.");
    }
  }

  ofport->send(ef);
  omport->send(interpolant);
}

    

CompileInfoHandle
ChangeFieldBasisAlgoCreate::get_compile_info(const TypeDescription *field_td)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldBasisAlgoCreateT");
  static const string base_class_name("ChangeFieldBasisAlgoCreate");

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


