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
 *  TransformVectorData: Unary field data operations
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformVectorData.h>

#include <iostream>

namespace SCIRun {

class TransformVectorData : public Module {
  GuiString functionx_;
  GuiString functiony_;
  GuiString functionz_;
  GuiInt    pre_normalize_;
  GuiInt    post_normalize_;
public:
  TransformVectorData(GuiContext* ctx);
  virtual ~TransformVectorData();
  virtual void execute();
};


DECLARE_MAKER(TransformVectorData)


TransformVectorData::TransformVectorData(GuiContext* ctx)
  : Module("TransformVectorData", ctx, Filter,"FieldsData", "SCIRun"),
    functionx_(ctx->subVar("functionx")),
    functiony_(ctx->subVar("functiony")),
    functionz_(ctx->subVar("functionz")),
    pre_normalize_(ctx->subVar("pre_normalize")),
    post_normalize_(ctx->subVar("post_normalize"))
{
}


TransformVectorData::~TransformVectorData()
{
}


void
TransformVectorData::execute()
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
  
  if (ifieldhandle->query_vector_interface(this).get_rep() == 0)
  {
    error("This module only works on vector fields.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfoHandle ci = TransformVectorDataAlgo::get_compile_info(ftd, ltd);
  Handle<TransformVectorDataAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldhandle;

  Function *functionx = new Function(3);
  Function *functiony = new Function(3);
  Function *functionz = new Function(3);
  fnparsestring(functionx_.get().c_str(), &functionx);
  fnparsestring(functiony_.get().c_str(), &functiony);
  fnparsestring(functionz_.get().c_str(), &functionz);
  ofieldhandle=algo->execute(ifieldhandle, functionx, functiony, functionz,
			     pre_normalize_.get(), post_normalize_.get());
  delete functionx;
  delete functiony;
  delete functionz;

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformVectorDataAlgo::get_compile_info(const TypeDescription *field_td,
					  const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TransformVectorDataAlgoT");
  static const string base_class_name("TransformVectorDataAlgo");

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
