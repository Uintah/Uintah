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
 *  TransformScalarData: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformScalarData.h>

#include <iostream>

namespace SCIRun {

class TransformScalarData : public Module {
  GuiString method_;
  GuiString function_;
  GuiDouble imin_;
  GuiDouble imax_;
  GuiDouble omin_;
  GuiDouble omax_;
public:
  TransformScalarData(GuiContext* ctx);
  virtual ~TransformScalarData();
  virtual void execute();
};


DECLARE_MAKER(TransformScalarData)


TransformScalarData::TransformScalarData(GuiContext* ctx)
  : Module("TransformScalarData", ctx, Filter,"FieldsData", "SCIRun"),
    method_(ctx->subVar("method")),
    function_(ctx->subVar("function")),
    imin_(ctx->subVar("imin")),
    imax_(ctx->subVar("imax")),
    omin_(ctx->subVar("omin")),
    omax_(ctx->subVar("omax"))
{
}


TransformScalarData::~TransformScalarData()
{
}


void
TransformScalarData::execute()
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
  
  ScalarFieldInterfaceHandle sfi;
  if ((sfi = ifieldhandle->query_scalar_interface(this)).get_rep() == 0)
  {
    error("This module only works on scalar fields.");
    return;
  }

  pair<double, double> minmax;
  sfi->compute_min_max(minmax.first, minmax.second);
  double imin=minmax.first;
  double imax=minmax.second;
  imin_.set(imin);
  imax_.set(imax);
  
  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfoHandle ci = TransformScalarDataAlgo::get_compile_info(ftd, ltd);
  Handle<TransformScalarDataAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldhandle;
  if (method_.get() == "function") {
    Function *function = new Function(1);
    fnparsestring(function_.get().c_str(), &function);
    ofieldhandle=algo->execute(ifieldhandle, function);
    delete function;
  } else if (method_.get() == "minmax") {
    double omin=omin_.get();
    double omax=omax_.get();
    if ((imin == imax) && (omin != omax)) {
      error("Input min/max are equal, unable to rescale");
      return;
    }
    double scale=1.0;
    if (imin != imax) 
      scale=(omax-omin)/(imax-imin);
    double trans=omin-(imin*scale);
    ofieldhandle=algo->execute(ifieldhandle, scale, trans);
  } else {
    error("Unrecognized method");
    return;
  }
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformScalarDataAlgo::get_compile_info(const TypeDescription *field_td,
					  const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TransformScalarDataAlgoT");
  static const string base_class_name("TransformScalarDataAlgo");

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
