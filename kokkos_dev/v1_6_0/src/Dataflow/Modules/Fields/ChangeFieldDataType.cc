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


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataType.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Datatypes/DenseMatrix.h>
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

  bool			check_types(FieldHandle);
  bool			types_equal_p(FieldHandle);
  virtual void		execute();
  virtual void		tcl_command(GuiArgs&, void*);
  GuiString		typename_;   // the out field type
  int			generation_;
};

  DECLARE_MAKER(ChangeFieldDataType)

ChangeFieldDataType::ChangeFieldDataType(GuiContext* ctx)
  : Module("ChangeFieldDataType", ctx, Source, "Fields", "SCIRun"),
    typename_(ctx->subVar("typename2")),
    generation_(-1)
{
}

ChangeFieldDataType::~ChangeFieldDataType(){
}



bool ChangeFieldDataType::check_types(FieldHandle f)
{
  const string &oname = f->get_type_description()->get_name();
  const string &nname = typename_.get();
  
  string::size_type oindx = oname.find('<');
  string::size_type nindx = nname.find('<');

  if (oindx == nindx)
  {
    if (oname.substr(0, oindx) == nname.substr(0, nindx))
    {
      return true;
    }
  }
  warning("The selected type and the input field type are incompatable.");
  return false;
}

bool ChangeFieldDataType::types_equal_p(FieldHandle f)
{
  const string &oname = f->get_type_description()->get_name();
  const string &nname = typename_.get();
  return oname == nname;
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
    gui->execute(string("set ")+id+"-typename \"---\"");
    return;
  }

  // The output port is required.
  FieldOPort *oport = (FieldOPort*)get_oport("Output Field");
  if (!oport) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  if (generation_ != fh.get_rep()->generation) 
  {
    generation_ = fh.get_rep()->generation;
    const string &tname = fh->get_type_description()->get_name();
    gui->execute(string("set ")+id+"-typename \"" + tname + "\"");
    gui->execute(id+" copy_attributes; update idletasks");
  }

  if (types_equal_p(fh))
  {
    // no changes, just send the original through (it may be nothing!)
    oport->send(fh);
    remark("Passing field from input port to output port unchanged.");
    return;
  }

  // verify that the requested edits are possible (type check)
  if (!check_types(fh))
  {
    typename_.set(fh->get_type_description()->get_name());
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = fh->get_type_description();
  CompileInfo *ci =
    ChangeFieldDataTypeAlgoCreate::get_compile_info(fsrc_td, typename_.get());
  Handle<ChangeFieldDataTypeAlgoCreate> algo;
  if (!module_dynamic_compile(*ci, algo)) return;

  gui->execute(id + " set_state Executing 0");
  bool same_value_type_p = false;
  FieldHandle ef(algo->execute(fh, fh->data_at(), same_value_type_p));


  // Copy over the field data
  const bool both_scalar_p =
    ef->query_scalar_interface() && fh->query_scalar_interface();
  if (both_scalar_p || same_value_type_p)
  {
    const TypeDescription *fdst_td = ef->get_type_description();
    CompileInfo *ci =
      ChangeFieldDataTypeAlgoCopy::get_compile_info(fsrc_td, fdst_td);
    Handle<ChangeFieldDataTypeAlgoCopy> algo;
    if (!module_dynamic_compile(*ci, algo)) return;

    gui->execute(id + " set_state Executing 0");
    algo->execute(fh, ef);
  }
  

  ScalarFieldInterface* sfi = ef->query_scalar_interface();
  if (sfi)
  {
    ef->freeze();
    std::pair<double, double> minmax(1, 0);
    sfi->compute_min_max(minmax.first, minmax.second);
    ef->set_property(string("minmax"), minmax, true);
  }
    
  oport->send(ef);
}

    
void ChangeFieldDataType::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("ChangeFieldDataType needs a minor command");
    return;
  }
 
  if (args[1] == "execute" || args[1] == "update_widget") {
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}


CompileInfo *
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


CompileInfo *
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


