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
 *  DirectInterpolate.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/DirectInterpolate.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class DirectInterpolate : public Module
{
  FieldIPort *src_port_;
  FieldIPort *dst_port_;
  FieldOPort *ofp_; 
  GuiInt     use_interp_;
  GuiInt     use_closest_;
  GuiDouble  closeness_distance_;

public:
  DirectInterpolate(GuiContext* ctx);
  virtual ~DirectInterpolate();
  virtual void execute();

  template <class Fld> void callback1(Fld *fld);

};

DECLARE_MAKER(DirectInterpolate)
DirectInterpolate::DirectInterpolate(GuiContext* ctx) : 
  Module("DirectInterpolate", ctx, Filter, "Fields", "SCIRun"),
  use_interp_(ctx->subVar("use_interp")),
  use_closest_(ctx->subVar("use_closest")),
  closeness_distance_(ctx->subVar("closeness_distance"))
{
}

DirectInterpolate::~DirectInterpolate()
{
}


void
DirectInterpolate::execute()
{
  dst_port_ = (FieldIPort *)get_iport("Destination");
  FieldHandle dfieldhandle;
  if (!dst_port_) {
    error("Unable to initialize iport 'Destination'.");
    return;
  }
  if (!(dst_port_->get(dfieldhandle) && dfieldhandle.get_rep()))
  {
    return;
  }

  src_port_ = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  if (!src_port_) {
    error("Unable to initialize iport 'Source'.");
    return;
  }
  if (!(src_port_->get(sfieldhandle) && sfieldhandle.get_rep()))
  {
    return;
  }

  ScalarFieldInterface *sfi = sfieldhandle->query_scalar_interface();
  VectorFieldInterface *vfi = sfieldhandle->query_vector_interface();
  TensorFieldInterface *tfi = sfieldhandle->query_tensor_interface();
  FieldHandle ofieldhandle;
  if (sfi)
  {
    const TypeDescription *td0 = dfieldhandle->get_type_description();
    const TypeDescription *td1 = dfieldhandle->data_at_type_description();
    CompileInfo *ci = DirectInterpScalarAlgoBase::get_compile_info(td0, td1);
    Handle<DirectInterpScalarAlgoBase> algo;
    if (!module_dynamic_compile(*ci, algo)) return;
    ofieldhandle = algo->execute(dfieldhandle, sfi,
				 use_interp_.get(),
				 use_closest_.get(),
				 closeness_distance_.get());
  }
  else if (vfi)
  {
    const TypeDescription *td0 = dfieldhandle->get_type_description();
    const TypeDescription *td1 = dfieldhandle->data_at_type_description();
    CompileInfo *ci = DirectInterpVectorAlgoBase::get_compile_info(td0, td1);
    Handle<DirectInterpVectorAlgoBase> algo;
    if (!module_dynamic_compile(*ci, algo)) return;
    ofieldhandle = algo->execute(dfieldhandle, vfi,
				 use_interp_.get(),
				 use_closest_.get(),
				 closeness_distance_.get());
  }
  else if (tfi)
  {
    const TypeDescription *td0 = dfieldhandle->get_type_description();
    const TypeDescription *td1 = dfieldhandle->data_at_type_description();
    CompileInfo *ci = DirectInterpTensorAlgoBase::get_compile_info(td0, td1);
    Handle<DirectInterpTensorAlgoBase> algo;
    if (!module_dynamic_compile(*ci, algo)) return;

    ofieldhandle = algo->execute(dfieldhandle, tfi,
				 use_interp_.get(),
				 use_closest_.get(),
				 closeness_distance_.get());
  }
  else
  {
    error("No field interface to sample on.");
  }

  ofp_ = (FieldOPort *)get_oport("Interpolant");
  if (!ofp_) {
    error("Unable to initialize " + name + "'s output port.");
    return;
  }
  string units;
  if (sfieldhandle->get_property("units", units))
    ofieldhandle->set_property("units", units, false);
  ofp_->send(ofieldhandle);
}



CompileInfo *
DirectInterpScalarAlgoBase::get_compile_info(const TypeDescription *field_td,
					     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectInterpScalarAlgo");
  static const string base_class_name("DirectInterpScalarAlgoBase");

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


CompileInfo *
DirectInterpVectorAlgoBase::get_compile_info(const TypeDescription *field_td,
					     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectInterpVectorAlgo");
  static const string base_class_name("DirectInterpVectorAlgoBase");

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


CompileInfo *
DirectInterpTensorAlgoBase::get_compile_info(const TypeDescription *field_td,
					     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectInterpTensorAlgo");
  static const string base_class_name("DirectInterpTensorAlgoBase");

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
