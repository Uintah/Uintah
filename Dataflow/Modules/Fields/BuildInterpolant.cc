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
 *  BuildInterpolant.cc:  Build an interpolant field -- a field that says
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
#include <Dataflow/Modules/Fields/BuildInterpolant.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class BuildInterpolant : public Module
{
  FieldIPort *src_port;
  FieldIPort *dst_port;
  FieldOPort *ofp; 
  GuiString   interp_op_gui_;

public:
  BuildInterpolant(const string& id);
  virtual ~BuildInterpolant();
  virtual void execute();

  //template <class Mesh, class Index>
  //void find_closest(Mesh *mesh, typename Index::index_type &idx, Point &p);
};

extern "C" Module* make_BuildInterpolant(const string& id)
{
  return new BuildInterpolant(id);
}

BuildInterpolant::BuildInterpolant(const string& id) : 
  Module("BuildInterpolant", id, Filter, "Fields", "SCIRun"),
  interp_op_gui_("interp_op_gui", id, this)
{
}

BuildInterpolant::~BuildInterpolant()
{
}



void
BuildInterpolant::execute()
{
  dst_port = (FieldIPort *)get_iport("Destination");
  FieldHandle fdst_h;

  if (!dst_port) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(dst_port->get(fdst_h) && fdst_h.get_rep()))
  {
    return;
  }

  src_port = (FieldIPort *)get_iport("Source");
  FieldHandle fsrc_h;
  if(!src_port) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(src_port->get(fsrc_h) && fsrc_h.get_rep()))
  {
    return;
  }

  CompileInfo *ci =
    BuildInterpAlgo::get_compile_info(fsrc_h->mesh()->get_type_description(),
				      fsrc_h->data_at_type_description(),
				      fdst_h->mesh()->get_type_description(),
				      fdst_h->data_at_type_description(),
				      fdst_h->get_type_description());
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    error("Could not compile algorithm.");
    return;
  }
  BuildInterpAlgo *algo =
    dynamic_cast<BuildInterpAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    error("Could not get algorithm.");
    return;
  }

  ofp = (FieldOPort *)get_oport("Interpolant");
  if(!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofp->send(algo->execute(fsrc_h->mesh(), fdst_h->mesh(), fdst_h->data_at()));
}



CompileInfo *
BuildInterpAlgo::get_compile_info(const TypeDescription *msrc,
				  const TypeDescription *lsrc,
				  const TypeDescription *mdst,
				  const TypeDescription *ldst,
				  const TypeDescription *fdst)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("BuildInterpAlgoT");
  static const string base_class_name("BuildInterpAlgo");

  const string::size_type loc = fdst->get_name().find_first_of('<');
  const string fout = fdst->get_name().substr(0, loc) +
    "<vector<pair<" + lsrc->get_name() + "::index_type, double> > > ";

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + "." +
		       lsrc->get_filename() + "." +
		       fdst->get_filename() + "." +
		       ldst->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name() + ", " +
                       lsrc->get_name() + ", " +
                       mdst->get_name() + ", " +
                       ldst->get_name() + ", " +
                       fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  fdst->fill_compile_info(rval);
  return rval;
}




} // End namespace SCIRun
