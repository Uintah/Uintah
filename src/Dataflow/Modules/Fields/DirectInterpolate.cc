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
  GuiString   interp_op_gui_;

public:
  DirectInterpolate(const string& id);
  virtual ~DirectInterpolate();
  virtual void execute();

  template <class Fld> void callback1(Fld *fld);

};

extern "C" Module* make_DirectInterpolate(const string& id)
{
  return new DirectInterpolate(id);
}

DirectInterpolate::DirectInterpolate(const string& id) : 
  Module("DirectInterpolate", id, Filter, "Fields", "SCIRun"),
  interp_op_gui_("interp_op_gui", id, this)
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
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(dst_port_->get(dfieldhandle) && dfieldhandle.get_rep()))
  {
    return;
  }

  src_port_ = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  if (!src_port_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(src_port_->get(sfieldhandle) && sfieldhandle.get_rep()))
  {
    return;
  }

  ScalarFieldInterface *sfi = sfieldhandle->query_scalar_interface();
  VectorFieldInterface *vfi = sfieldhandle->query_vector_interface();
  FieldHandle ofieldhandle;
  if (sfi)
  {
    const TypeDescription *td0 = dfieldhandle->get_type_description();
    const TypeDescription *td1 = dfieldhandle->data_at_type_description();
    CompileInfo *ci = DirectInterpScalarAlgoBase::get_compile_info(td0, td1);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }

    DirectInterpScalarAlgoBase *algo = 
      dynamic_cast<DirectInterpScalarAlgoBase *>(algo_handle.get_rep());
    if (algo == 0) 
    {
      error("Could not get algorithm.");
      return;
    }
    ofieldhandle = algo->execute(dfieldhandle, sfi);
  }
  else if (vfi)
  {
    const TypeDescription *td0 = dfieldhandle->get_type_description();
    const TypeDescription *td1 = dfieldhandle->data_at_type_description();
    CompileInfo *ci = DirectInterpVectorAlgoBase::get_compile_info(td0, td1);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }

    DirectInterpVectorAlgoBase *algo = 
      dynamic_cast<DirectInterpVectorAlgoBase *>(algo_handle.get_rep());
    if (algo == 0) 
    {
      error("Could not get algorithm.");
      return;
    }
    ofieldhandle = algo->execute(dfieldhandle, vfi);
  }
  else
  {
    error("No scalar or vector field interface to sample on.");
  }

  ofp_ = (FieldOPort *)get_oport("Interpolant");
  if (!ofp_) {
    error("Unable to initialize " + name + "'s output port.");
    return;
  }
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


} // End namespace SCIRun
