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

/*
 *  ReplaceScalarDataValue: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/ReplaceScalarDataValue.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class ReplaceScalarDataValue : public Module {
  GuiDouble oldvalue_;
  GuiDouble newvalue_;
public:
  ReplaceScalarDataValue(GuiContext* ctx);
  virtual ~ReplaceScalarDataValue();
  virtual void execute();
};


DECLARE_MAKER(ReplaceScalarDataValue)


  ReplaceScalarDataValue::ReplaceScalarDataValue(GuiContext* ctx)
    : Module("ReplaceScalarDataValue", ctx, Filter,"FieldsData", "SCIRun"),
      oldvalue_(ctx->subVar("oldvalue")), newvalue_(ctx->subVar("newvalue"))
{
}


ReplaceScalarDataValue::~ReplaceScalarDataValue()
{
}


void
ReplaceScalarDataValue::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Input field is empty.");
    return;
  }
  
  if (ifieldhandle->query_scalar_interface(this).get_rep() == 0)
  {
    error("This module only works on scalar fields.");
    return;
  }

  double oldvalue = oldvalue_.get();
  double newvalue = newvalue_.get();

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfoHandle ci =
    ReplaceScalarDataValueAlgo::get_compile_info(ftd, ltd);
  Handle<ReplaceScalarDataValueAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle, oldvalue, newvalue));

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
ReplaceScalarDataValueAlgo::get_compile_info(const TypeDescription *field_td,
					 const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ReplaceScalarDataValueAlgoT");
  static const string base_class_name("ReplaceScalarDataValueAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name() + ", " + loc_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
