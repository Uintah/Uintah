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
 *   David Weinstein
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/DirectInterpolate.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class DirectInterpolate : public Module
{
  FieldIPort *src_port;
  FieldIPort *dst_port;
  FieldOPort *ofp; 
  GuiString  interpolation_basis_;
  GuiInt     map_source_to_single_dest_;
  GuiInt     exhaustive_search_;
  GuiDouble  exhaustive_search_max_dist_;
  GuiInt     np_;

public:
  DirectInterpolate(GuiContext* ctx);
  virtual ~DirectInterpolate();
  virtual void execute();
};

DECLARE_MAKER(DirectInterpolate)
DirectInterpolate::DirectInterpolate(GuiContext* ctx) : 
  Module("DirectInterpolate", ctx, Filter, "FieldsData", "SCIRun"),
  interpolation_basis_(ctx->subVar("interpolation_basis")),
  map_source_to_single_dest_(ctx->subVar("map_source_to_single_dest")),
  exhaustive_search_(ctx->subVar("exhaustive_search")),
  exhaustive_search_max_dist_(ctx->subVar("exhaustive_search_max_dist")),
  np_(ctx->subVar("np"))
{
}

DirectInterpolate::~DirectInterpolate()
{
}

void
DirectInterpolate::execute()
{
  dst_port = (FieldIPort *)get_iport("Destination");
  FieldHandle fdst_h;

  if (!dst_port) {
    error("Unable to initialize iport 'Destination'.");
    return;
  }
  if (!(dst_port->get(fdst_h) && fdst_h.get_rep()))
  {
    return;
  }
  if (fdst_h->data_at() == Field::NONE)
  {
    warning("No data location in destination to interpolate to.");
    return;
  }

  src_port = (FieldIPort *)get_iport("Source");
  FieldHandle fsrc_h;
  if(!src_port) {
    error("Unable to initialize iport 'Source'.");
    return;
  }
  if (!(src_port->get(fsrc_h) && fsrc_h.get_rep()))
  {
    return;
  }
  if (!fsrc_h->data_at() == Field::NONE)
  {
    warning("No data location in Source field to interpolate from.");
    return;
  }

  CompileInfoHandle ci =
    DirectInterpAlgo::get_compile_info(fsrc_h->get_type_description(),
				       fsrc_h->data_at_type_description(),
				       fdst_h->get_type_description(),
				       fdst_h->data_at_type_description());
  Handle<DirectInterpAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  ofp = (FieldOPort *)get_oport("Interpolant");
  if(!ofp) {
    error("Unable to initialize oport 'Interpolant'.");
    return;
  }
  fsrc_h->mesh()->synchronize(Mesh::LOCATE_E);
  ofp->send(algo->execute(fsrc_h, fdst_h->mesh(), fdst_h->data_at(),
			  interpolation_basis_.get(),
			  map_source_to_single_dest_.get(),
			  exhaustive_search_.get(),
			  exhaustive_search_max_dist_.get(), np_.get()));
}

CompileInfoHandle
DirectInterpAlgo::get_compile_info(const TypeDescription *fsrc,
				   const TypeDescription *lsrc,
				   const TypeDescription *fdst,
				   const TypeDescription *ldst)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectInterpAlgoT");
  static const string base_class_name("DirectInterpAlgo");

  const string::size_type fdst_loc = fdst->get_name().find_first_of('<');
  const string::size_type fsrc_loc = fsrc->get_name().find_first_of('<');
  const string fout = fdst->get_name().substr(0, fdst_loc) +
    fsrc->get_name().substr(fsrc_loc);

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       lsrc->get_filename() + "." +
		       fdst->get_filename() + "." +
		       ldst->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + ", " +
                       lsrc->get_name() + ", " +
                       fout + ", " +
                       ldst->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  fdst->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
