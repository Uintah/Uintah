/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Containers/Handle.h>
#include <Core/Util/DynamicCompilation.h>
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
  GuiString  interpolation_basis_;
  GuiInt     map_source_to_single_dest_;
  GuiInt     exhaustive_search_;
  GuiDouble  exhaustive_search_max_dist_;
  GuiInt     np_;

public:
  BuildInterpolant(GuiContext* ctx);
  virtual ~BuildInterpolant();
  virtual void execute();
};

DECLARE_MAKER(BuildInterpolant)
BuildInterpolant::BuildInterpolant(GuiContext* ctx) : 
  Module("BuildInterpolant", ctx, Filter, "FieldsData", "SCIRun"),
  interpolation_basis_(ctx->subVar("interpolation_basis")),
  map_source_to_single_dest_(ctx->subVar("map_source_to_single_dest")),
  exhaustive_search_(ctx->subVar("exhaustive_search")),
  exhaustive_search_max_dist_(ctx->subVar("exhaustive_search_max_dist")),
  np_(ctx->subVar("np"))
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
    error("Unable to initialize iport 'Destination'.");
    return;
  }
  if (!(dst_port->get(fdst_h) && fdst_h.get_rep()))
  {
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

  CompileInfoHandle ci =
    BuildInterpAlgo::get_compile_info(fsrc_h->mesh()->get_type_description(),
				      fsrc_h->order_type_description(),
				      fdst_h->mesh()->get_type_description(),
				      fdst_h->order_type_description(),
				      fdst_h->get_type_description());
  Handle<BuildInterpAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  ofp = (FieldOPort *)get_oport("Interpolant");
  if(!ofp) {
    error("Unable to initialize oport 'Interpolant'.");
    return;
  }
  fsrc_h->mesh()->synchronize(Mesh::LOCATE_E);
  ofp->send(algo->execute(fsrc_h->mesh(), fdst_h->mesh(), fdst_h->basis_order(),
			  interpolation_basis_.get(),
			  map_source_to_single_dest_.get(),
			  exhaustive_search_.get(),
			  exhaustive_search_max_dist_.get(), np_.get()));
}

CompileInfoHandle
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
		       lsrc->get_filename() + "." +
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
