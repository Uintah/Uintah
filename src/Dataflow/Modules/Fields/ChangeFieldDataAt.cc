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

//    File   : ChangeFieldDataAt.cc
//    Author : McKay Davis
//    Date   : July 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/share/share.h>

#include <Core/Containers/Handle.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataAt.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class PSECORESHARE ChangeFieldDataAt : public Module {
public:
  GuiString outputdataat_;     // the out data at
  int              generation_;
  ChangeFieldDataAt(GuiContext* ctx);
  virtual ~ChangeFieldDataAt();
  virtual void execute();
  void update_input_attributes(FieldHandle);
};

  DECLARE_MAKER(ChangeFieldDataAt)

ChangeFieldDataAt::ChangeFieldDataAt(GuiContext* ctx)
  : Module("ChangeFieldDataAt", ctx, Filter, "Fields", "SCIRun"),
    outputdataat_(ctx->subVar("outputdataat")),
    generation_(-1)
{
}

ChangeFieldDataAt::~ChangeFieldDataAt(){
}


void
ChangeFieldDataAt::update_input_attributes(FieldHandle f) 
{
  switch(f->data_at())
  {
  case Field::NODE:
    gui->execute(string("set ")+id+"-inputdataat Nodes"); break;
  case Field::EDGE: 
    gui->execute(string("set ")+id+"-inputdataat Edges"); break;
  case Field::FACE: 
    gui->execute(string("set ")+id+"-inputdataat Faces"); break;
  case Field::CELL: 
    gui->execute(string("set ")+id+"-inputdataat Cells"); break;
  case Field::NONE: 
    gui->execute(string("set ")+id+"-inputdataat None"); break;
  default: ;
  }

  string fldname;
  if (f->get_property("name",fldname))
    gui->execute(string("set ")+id+"-fldname "+fldname);
  else
    gui->execute(string("set ")+id+"-fldname \"--- Name Not Assigned ---\"");

  // copy valid settings to the un-checked output field attributes
  gui->execute(id+" copy_attributes; update idletasks");
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
    gui->execute(string("set ")+id+"-inputdataat \"---\"");
    return;
  }

  if (generation_ != fh.get_rep()->generation)
  {
    update_input_attributes(fh);
    generation_ = fh.get_rep()->generation;
  }

  // The output port is required.
  FieldOPort *oport = (FieldOPort*)get_oport("Output Field");
  if (!oport) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  Field::data_location dataat = fh->data_at();
  const string &d = outputdataat_.get();
  if (d == "Nodes")
  {
    dataat = Field::NODE;
  }
  else if (d == "Edges")
  {
    dataat = Field::EDGE;
  }
  else if (d == "Faces")
  {
    dataat = Field::FACE;
  }
  else if (d == "Cells")
  {
    dataat = Field::CELL;
  }
  else if (d == "None")
  {
    dataat = Field::NONE;
  }

  if (dataat == fh->data_at())
  {
    // no changes, just send the original through (it may be nothing!)
    oport->send(fh);
    remark("Passing field from input port to output port unchanged.");
    return;
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = fh->get_type_description();
  CompileInfoHandle ci = ChangeFieldDataAtAlgoCreate::get_compile_info
    (fsrc_td, fh->get_type_description()->get_name());
  Handle<ChangeFieldDataAtAlgoCreate> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  gui->execute(id + " set_state Executing 0");
  bool same_value_type_p = false;
  FieldHandle ef(algo->execute(fh, dataat, same_value_type_p));

  oport->send(ef);
}

    

CompileInfoHandle
ChangeFieldDataAtAlgoCreate::get_compile_info(const TypeDescription *field_td,
					      const string &fdstname)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("ChangeFieldDataAtAlgoCreateT");
  static const string base_class_name("ChangeFieldDataAtAlgoCreate");

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

} // End namespace Moulding


