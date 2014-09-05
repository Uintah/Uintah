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

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Packages/BioPSE/Dataflow/Modules/Forward/TensorsToIndices.h>
#include <Core/Geometry/Tensor.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class TensorsToIndices : public Module {
public:
  TensorsToIndices(GuiContext *context);
  virtual ~TensorsToIndices();
  virtual void execute();
};


DECLARE_MAKER(TensorsToIndices)


TensorsToIndices::TensorsToIndices(GuiContext *context)
  : Module("TensorsToIndices", context, Filter, "Forward", "BioPSE")
{
}

TensorsToIndices::~TensorsToIndices()
{
}

void TensorsToIndices::execute() {
  FieldIPort* ifieldport = (FieldIPort *) get_iport("IndexField");
  FieldOPort* ofieldport = (FieldOPort *) get_oport("TensorField");
  FieldHandle ifieldH;
  if (!ifieldport->get(ifieldH))
    return;
  if (!ifieldH.get_rep()) {
    error("Empty input field.");
    return;
  }

  const TypeDescription *field_src_td = ifieldH->get_type_description();
  const string field_dst_name = 
    ifieldH->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
    ifieldH->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
    ifieldH->get_type_description(Field::BASIS_TD_E)->get_similar_name("int", 
                                                       0, "<", " >, ") +
    ifieldH->get_type_description(Field::FDATA_TD_E)->get_similar_name("int",
                                                       0, "<", " >") + " >";
  
  CompileInfoHandle ci =
    TensorsToIndicesAlgo::get_compile_info(field_src_td, field_dst_name);
  Handle<TensorsToIndicesAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldH = algo->execute(ifieldH);
  ofieldport->send_and_dereference(ofieldH);
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
