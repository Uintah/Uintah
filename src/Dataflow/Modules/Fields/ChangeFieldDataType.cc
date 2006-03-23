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

using std::endl;
using std::pair;

class ChangeFieldDataType : public Module {
public:
  ChangeFieldDataType(GuiContext* ctx);
  virtual ~ChangeFieldDataType();

  bool			types_equal_p(FieldHandle);
  virtual void		execute();

  GuiString		outputdatatype_;   // the out field type
  GuiString		inputdatatype_;    // the input field type
  GuiString		fldname_;          // the input field name
  int			generation_;
  string                last_data_str_;
  FieldHandle           outputfield_;
};

DECLARE_MAKER(ChangeFieldDataType)

ChangeFieldDataType::ChangeFieldDataType(GuiContext* ctx)
  : Module("ChangeFieldDataType", ctx, Filter, "FieldsData", "SCIRun"),
    outputdatatype_(get_ctx()->subVar("outputdatatype"), "double"),
    inputdatatype_(get_ctx()->subVar("inputdatatype", false), "---"),
    fldname_(get_ctx()->subVar("fldname", false), "---"),
    generation_(-1),
    outputfield_(0)
{
}

ChangeFieldDataType::~ChangeFieldDataType()
{
  fldname_.set("---");
  inputdatatype_.set("---");
}



bool ChangeFieldDataType::types_equal_p(FieldHandle f)
{
  const string &iname = f->get_type_description()->get_name();
  const string &oname = outputdatatype_.get();
  return iname == oname;
}


void
ChangeFieldDataType::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 

  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    fldname_.set("---");
    inputdatatype_.set("---");
    return;
  }

  // The output port is required.
  FieldOPort *oport = (FieldOPort*)get_oport("Output Field");

  const string old_data_str = fh->get_type_description(Field::MESH_TD_E)->get_name();
  const string new_data_str = outputdatatype_.get();

  if (generation_ != fh.get_rep()->generation) 
  {
    generation_ = fh.get_rep()->generation;
    const string &tname = fh->get_type_description()->get_name();
    inputdatatype_.set(tname);

    string fldname;
    if (fh->get_property("name",fldname))
    {
      fldname_.set(fldname);
    }
    else
    {
      fldname_.set("--- No Name ---");
    }
  }
  else if (new_data_str == last_data_str_ && oport_cached("Output Field"))
  {
    oport->send_and_dereference(outputfield_, true);
    return;
  }
  last_data_str_ = new_data_str;

  if (old_data_str == new_data_str)
  {
    // No changes, just send the original through.
    outputfield_ = fh;
    remark("Passing field from input port to output port unchanged.");
    oport->send_and_dereference(outputfield_, true);
    return;
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = fh->get_type_description();
    const string oftn = 
      fh->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      fh->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      fh->get_type_description(Field::BASIS_TD_E)->get_similar_name(new_data_str,
							 0, "<", " >, ") +
      fh->get_type_description(Field::FDATA_TD_E)->get_similar_name(new_data_str,
							 0, "<", " >") + " >";

  CompileInfoHandle create_ci =
    ChangeFieldDataTypeAlgoCreate::get_compile_info(fsrc_td, oftn);
  Handle<ChangeFieldDataTypeAlgoCreate> create_algo;
  if (!DynamicCompilation::compile(create_ci, create_algo, this))
  {
    error("Unable to compile creation algorithm.");
    return;
  }
  update_state(Executing);
  outputfield_ = create_algo->execute(fh);

  if (fh->basis_order() != -1)
  {
    const TypeDescription *fdst_td = outputfield_->get_type_description();
    CompileInfoHandle copy_ci =
      ChangeFieldDataTypeAlgoCopy::get_compile_info(fsrc_td, fdst_td);
    Handle<ChangeFieldDataTypeAlgoCopy> copy_algo;
    
    if (new_data_str == "Vector" && 
	fh->query_scalar_interface(this).get_rep() ||
	!DynamicCompilation::compile(copy_ci, copy_algo, true, this))
    {
      warning("Unable to convert the old data from " + old_data_str +
	      " to " + new_data_str + ", no data transfered.");
    }
    else
    {
      remark("Copying " + old_data_str + " data into " + new_data_str +
	     " may result in a loss of precision.");
      update_state(Executing);
      copy_algo->execute(fh, outputfield_);
    }
  }
    
  oport->send_and_dereference(outputfield_, true);
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


