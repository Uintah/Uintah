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


//    File   : ChangeFieldDataType.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Core/Containers/Handle.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataType.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

class ChangeFieldDataType : public Module {

public:
  ChangeFieldDataType(GuiContext* ctx);
  virtual ~ChangeFieldDataType();
  virtual void execute();

private:

  FieldHandle  field_output_handle_;


  GuiString		gui_field_name_;        // the input field name
  GuiString		gui_input_datatype_;    // the input field type
  GuiString		gui_output_datatype_;   // the output field type

  string                last_data_str_;

  FieldHandle           outputfield_;
};

DECLARE_MAKER(ChangeFieldDataType)

ChangeFieldDataType::ChangeFieldDataType(GuiContext* ctx)
  : Module("ChangeFieldDataType", ctx, Filter, "FieldsData", "SCIRun"),

    field_output_handle_(0),

    gui_field_name_(get_ctx()->subVar("field_name", false), "---"),

    gui_input_datatype_(get_ctx()->subVar("input_datatype", false), "---"),

    gui_output_datatype_(get_ctx()->subVar("output_datatype"), "double")
{
}


ChangeFieldDataType::~ChangeFieldDataType()
{
}


void
ChangeFieldDataType::execute()
{
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input Field", field_input_handle, true ) ) {
    gui_field_name_.set("---");
    gui_input_datatype_.set("---");
    return;
  }

  string old_datatype_str =
    field_input_handle->get_type_description(Field::FDATA_TD_E)->get_name();

  std::string::size_type pos = old_datatype_str.find( "<" );

  if( pos != string::npos )
    old_datatype_str.erase(0, pos+1 );
  
  pos = old_datatype_str.find(",");
  if( pos != std::string::npos )
    old_datatype_str.erase( pos, old_datatype_str.length()-pos );

  // Check to see if the input field has changed.
  if( inputs_changed_ ) {
    string fldname;
    if (field_input_handle->get_property("name",fldname))
      gui_field_name_.set(fldname);
    else
      gui_field_name_.set("--- No Name ---");

    gui_input_datatype_.set(old_datatype_str);
  }

  if( gui_output_datatype_.changed( true ) ||
      !field_output_handle_.get_rep() ||
      inputs_changed_ ) {

    update_state(Executing);

    const string new_datatype_str = gui_output_datatype_.get();

    if (new_datatype_str == old_datatype_str) {
      // No changes, just send the original field through.
      field_output_handle_ = field_input_handle;
      remark("Passing field from input port to output port unchanged.");

    } else {
      // Create a field identical to the input, except for the datatype.
      const TypeDescription *fsrc_td = field_input_handle->get_type_description();
      const string oftn = 
	field_input_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
	field_input_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
	field_input_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(new_datatype_str,
										      0, "<", " >, ") +
	field_input_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(new_datatype_str,
							 0, "<", " >") + " >";
      
      CompileInfoHandle create_ci =
	ChangeFieldDataTypeAlgoCreate::get_compile_info(fsrc_td, oftn);
      Handle<ChangeFieldDataTypeAlgoCreate> create_algo;
      if (!DynamicCompilation::compile(create_ci, create_algo, this)) {
	error("Unable to compile creation algorithm.");
	return;
      }

      field_output_handle_ = create_algo->execute(field_input_handle);

      if (field_input_handle->basis_order() != -1) {
	const TypeDescription *fdst_td = field_output_handle_->get_type_description();
	CompileInfoHandle copy_ci =
	  ChangeFieldDataTypeAlgoCopy::get_compile_info(fsrc_td, fdst_td);
	Handle<ChangeFieldDataTypeAlgoCopy> copy_algo;
	
	if (new_datatype_str == "Vector" && 
	field_input_handle->query_scalar_interface(this).get_rep() ||
	    !DynamicCompilation::compile(copy_ci, copy_algo, true, this)) {
	  warning("Unable to convert the old data from " + old_datatype_str +
		  " to " + new_datatype_str + ", no data transfered.");
	} else {
	  remark("Copying " + old_datatype_str + " data into " + new_datatype_str +
		 " may result in a loss of precision.");
	  update_state(Executing);
	  copy_algo->execute(field_input_handle, field_output_handle_);
	}
      }
    }
  }
    
  // Send the new field downstream
  send_output_handle("Output Field", field_output_handle_, true);
}

    
CompileInfoHandle
ChangeFieldDataTypeAlgoCreate::get_compile_info(const TypeDescription *ftd,
                                                const string &fdstname)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldDataTypeAlgoCreateT");
  static const string base_class_name("ChangeFieldDataTypeAlgoCreate");

  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       ftd->get_filename() + "." +
		       to_filename(fdstname) + ".",
		       base_class_name, 
		       template_class,
                       ftd->get_name() + "," + fdstname + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  rval->add_data_include("../src/Core/Geometry/Vector.h");
  rval->add_data_include("../src/Core/Geometry/Tensor.h");
  return rval;
}


CompileInfoHandle
ChangeFieldDataTypeAlgoCopy::get_compile_info(const TypeDescription *fsrctd,
                                              const TypeDescription *fdsttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldDataTypeAlgoCopyT");
  static const string base_class_name("ChangeFieldDataTypeAlgoCopy");

  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       fsrctd->get_filename() + "." +
		       fdsttd->get_filename() + ".",
                       base_class_name, 
		       template_class,
                       fsrctd->get_name() + "," + fdsttd->get_name() + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrctd->fill_compile_info(rval);
  return rval;
}


} // End namespace Moulding


