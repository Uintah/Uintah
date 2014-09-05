/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

//    File   : ChangeFieldDataType.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/share/share.h>

#include <Core/Containers/Handle.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataType.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class PSECORESHARE ChangeFieldDataType : public Module {
public:
  ChangeFieldDataType(GuiContext* ctx);
  virtual ~ChangeFieldDataType();

  bool			types_equal_p(FieldHandle);
  virtual void		execute();
  virtual void		tcl_command(GuiArgs&, void*);
  GuiString		outputdatatype_;   // the out field type
  GuiString		inputdatatype_;    // the input field type
  GuiString		fldname_;          // the input field name
  int			generation_;
  string                last_data_type_;
  FieldHandle           outputfield_;
};

DECLARE_MAKER(ChangeFieldDataType);

ChangeFieldDataType::ChangeFieldDataType(GuiContext* ctx)
  : Module("ChangeFieldDataType", ctx, Filter, "FieldsData", "SCIRun"),
    outputdatatype_(ctx->subVar("outputdatatype")),
    inputdatatype_(ctx->subVar("inputdatatype", false)),
    fldname_(ctx->subVar("fldname", false)),
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
  if (!iport) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  
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
  if (!oport) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  const string old_data_type = fh->get_type_description(1)->get_name();
  const string new_data_type = outputdatatype_.get();
  const string new_field_type =
    fh->get_type_description(0)->get_name() + "<" + new_data_type + "> ";

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
  else if (new_data_type == last_data_type_)
  {
    oport->send(outputfield_);
    return;
  }
  last_data_type_ = new_data_type;

  if (old_data_type == new_data_type)
  {
    // No changes, just send the original through.
    outputfield_ = fh;
    remark("Passing field from input port to output port unchanged.");
    oport->send(outputfield_);
    return;
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = fh->get_type_description();
  CompileInfoHandle create_ci =
    ChangeFieldDataTypeAlgoCreate::get_compile_info(fsrc_td, new_field_type);
  Handle<ChangeFieldDataTypeAlgoCreate> create_algo;
  if (!DynamicCompilation::compile(create_ci, create_algo, this))
  {
    error("Unable to compile creation algorithm.");
    return;
  }
  update_state(Executing);
  outputfield_ = create_algo->execute(fh);


  if (fh->data_at() != Field::NONE)
  {
    const TypeDescription *fdst_td = outputfield_->get_type_description();
    CompileInfoHandle copy_ci =
      ChangeFieldDataTypeAlgoCopy::get_compile_info(fsrc_td, fdst_td);
    Handle<ChangeFieldDataTypeAlgoCopy> copy_algo;

    if (new_data_type == "Vector" && 
	fh->query_scalar_interface(this).get_rep() ||
	!DynamicCompilation::compile(copy_ci, copy_algo, true, this))
    {
      warning("Unable to convert the old data from " + old_data_type +
	      " to " + new_data_type + ", no data transfered.");
    }
    else
    {
      remark("Copying " + old_data_type + " data into " + new_data_type +
	     " may result in a loss of precision.");
      update_state(Executing);
      copy_algo->execute(fh, outputfield_);
    }
  }

  oport->send(outputfield_);
}

    
void
ChangeFieldDataType::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2)
  {
    args.error("ChangeFieldDataType needs a minor command");
    return;
  }
 
  if (args[1] == "execute" || args[1] == "update_widget")
  {
    want_to_execute();
  }
  else
  {
    Module::tcl_command(args, userdata);
  }
}


CompileInfoHandle
ChangeFieldDataTypeAlgoCreate::get_compile_info(const TypeDescription *field_td,
				    const string &fdstname)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldDataTypeAlgoCreateT");
  static const string base_class_name("ChangeFieldDataTypeAlgoCreate");

  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       field_td->get_filename() + "." +
		       to_filename(fdstname) + ".",
		       base_class_name, 
		       template_class,
                       field_td->get_name() + "," + fdstname + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
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


