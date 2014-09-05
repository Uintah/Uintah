/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ConvertFieldDataFromIndicesToTensors: Change a Field of indices (ints) into a Field or Tensors,
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
#include <Packages/BioPSE/Dataflow/Modules/Forward/ConvertFieldDataFromIndicesToTensors.h>
#include <Core/Geometry/Tensor.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class ConvertFieldDataFromIndicesToTensors : public Module {
public:
  ConvertFieldDataFromIndicesToTensors(GuiContext *context);
  virtual ~ConvertFieldDataFromIndicesToTensors();
  virtual void execute();
};


DECLARE_MAKER(ConvertFieldDataFromIndicesToTensors)


ConvertFieldDataFromIndicesToTensors::ConvertFieldDataFromIndicesToTensors(GuiContext *context)
  : Module("ConvertFieldDataFromIndicesToTensors", context, Filter, "Forward", "BioPSE")
{
}


ConvertFieldDataFromIndicesToTensors::~ConvertFieldDataFromIndicesToTensors()
{
}


void
ConvertFieldDataFromIndicesToTensors::execute()
{
  FieldHandle ifieldH;
  if (!get_input_handle("IndexField", ifieldH)) return;

  vector<pair<string, Tensor> > conds;
  if (!ifieldH->get_property("conductivity_table", conds)) {
    error("Error - input field does not have a conductivity_table property.");
    return;
  }

  const TypeDescription *field_src_td = ifieldH->get_type_description();
  const string field_dst_name = 
    ifieldH->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
    ifieldH->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
    ifieldH->get_type_description(Field::BASIS_TD_E)->get_similar_name("Tensor", 
                                                       0, "<", " >, ") +
    ifieldH->get_type_description(Field::FDATA_TD_E)->get_similar_name("Tensor",
                                                       0, "<", " >") + " >";

  CompileInfoHandle ci =
    ConvertFieldDataFromIndicesToTensorsAlgo::get_compile_info(field_src_td, field_dst_name);
  Handle<ConvertFieldDataFromIndicesToTensorsAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldH = algo->execute(ifieldH);

  send_output_handle("TensorField", ofieldH);
}


} // End namespace BioPSE


namespace SCIRun {
CompileInfoHandle
ConvertFieldDataFromIndicesToTensorsAlgo::get_compile_info(const TypeDescription *field_src_td,
				       const string &field_dst_name)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ConvertFieldDataFromIndicesToTensorsAlgoT");
  static const string base_class_name("ConvertFieldDataFromIndicesToTensorsAlgo");

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
  rval->add_data_include("../src/Core/Geometry/Tensor.h");
  return rval;
}
} // End namespace SCIRun
