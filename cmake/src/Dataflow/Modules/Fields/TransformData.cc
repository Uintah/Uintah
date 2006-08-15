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
 *  TransformData: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformData.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <iostream>

namespace SCIRun {

class TransformData : public Module
{
public:
  TransformData(GuiContext* ctx);
  virtual ~TransformData();
  virtual void execute();

  virtual void presave();

private:
  GuiString gui_function_;
  GuiString gui_output_data_type_;
  
  FieldHandle field_output_handle_;
};


DECLARE_MAKER(TransformData)


TransformData::TransformData(GuiContext* ctx)
  : Module("TransformData", ctx, Filter,"FieldsData", "SCIRun"),
    gui_function_(get_ctx()->subVar("function"), "result = v * 10;"),
    gui_output_data_type_(get_ctx()->subVar("outputdatatype"), "input"),
    field_output_handle_(0)
{
}


TransformData::~TransformData()
{
}


void
TransformData::execute()
{
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input Field", field_input_handle, true ) ) return;

  if (field_input_handle->basis_order() == -1) {
    error("Field 0 contains no data to transform.");
    return;
  }
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before get.
  // Check to see if the input field has changed.
  if( inputs_changed_ ||
      gui_function_.changed( true ) ||
      gui_output_data_type_.changed( true ) ||
      !field_output_handle_.get_rep() )
  {
    update_state(Executing);

    string function = gui_function_.get();

    // Remove trailing white-space from the function string.
    while (function.size() && isspace(function[function.size()-1]))
      function.resize(function.size()-1);

    string outputDataType = gui_output_data_type_.get();

    if (outputDataType == "input") {
      TypeDescription::td_vec *tdv = 
			field_input_handle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      outputDataType = (*tdv)[0]->get_name();
    }

    const TypeDescription *ftd = field_input_handle->get_type_description();
    const TypeDescription *ltd = field_input_handle->order_type_description();
    const string oftn = 
      field_input_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      field_input_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      field_input_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(outputDataType, 
							 0, "<", " >, ") +
      field_input_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(outputDataType,
							 0, "<", " >") + " >";
    int hoffset = 0;
    Handle<TransformDataAlgo> algo;
    
    while (1) {
      CompileInfoHandle ci =
	TransformDataAlgo::get_compile_info(ftd, oftn, ltd, function, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this)) {
	error("Your function would not compile.");
	get_gui()->eval(get_id() + " compile_error "+ci->filename_);
	DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	return;
      }

      if (algo->identify() == function) {
	break;
      }
      hoffset++;
    }

    field_output_handle_ = algo->execute(field_input_handle);
  }

  send_output_handle( "Output Field",  field_output_handle_, true );
}


void
TransformData::presave()
{
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before saving.
}


CompileInfoHandle
TransformDataAlgo::get_compile_info(const TypeDescription *field_td,
				    string ofieldtypename,
				    const TypeDescription *loc_td,
				    string function,
				    int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("TransformDataInstance" + to_string(hashval));
  static const string base_class_name("TransformDataAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field_td->get_filename() + "." +
		       to_filename(ofieldtypename) + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       field_td->get_name() + ", " +
		       ofieldtypename + ", " +
		       loc_td->get_name());

  // Code for the function.
  string class_declaration =
    string("template <class IFIELD, class OFIELD, class LOC>\n") +
    "class " + template_name + " : public TransformDataAlgoT<IFIELD, OFIELD, LOC>\n" +
    "{\n" +
    "  virtual void function(typename OFIELD::value_type &result,\n" +
    "                        double x, double y, double z,\n" +
    "                        const typename IFIELD::value_type &v)\n" +
    "  {\n" +
    "    " + function + "\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  field_td->fill_compile_info(rval);
  rval->add_data_include("../src/Core/Geometry/Vector.h");
  rval->add_data_include("../src/Core/Geometry/Tensor.h");
  return rval;
}


} // End namespace SCIRun
