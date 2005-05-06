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
 *  BuildMappingMatrix.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   Jan 2005
 *
 *  Copyright (C) 2005 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/BuildMappingMatrix.h>
#include <Core/Containers/Handle.h>
#include <Core/Util/DynamicCompilation.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class BuildMappingMatrix : public Module
{
  GuiString  interpolation_basis_;
  GuiInt     map_source_to_single_dest_;
  GuiInt     exhaustive_search_;
  GuiDouble  exhaustive_search_max_dist_;
  GuiInt     np_;

public:
  BuildMappingMatrix(GuiContext* ctx);
  virtual ~BuildMappingMatrix();
  virtual void execute();
};

DECLARE_MAKER(BuildMappingMatrix)
BuildMappingMatrix::BuildMappingMatrix(GuiContext* ctx) : 
  Module("BuildMappingMatrix", ctx, Filter, "FieldsData", "SCIRun"),
  interpolation_basis_(ctx->subVar("interpolation_basis")),
  map_source_to_single_dest_(ctx->subVar("map_source_to_single_dest")),
  exhaustive_search_(ctx->subVar("exhaustive_search")),
  exhaustive_search_max_dist_(ctx->subVar("exhaustive_search_max_dist")),
  np_(ctx->subVar("np"))
{
}

BuildMappingMatrix::~BuildMappingMatrix()
{
}

void
BuildMappingMatrix::execute()
{
  FieldIPort *dst_port = (FieldIPort *)get_iport("Destination");
  FieldHandle fdst_h;
  if (!(dst_port->get(fdst_h) && fdst_h.get_rep()))
  {
    return;
  }
  if (fdst_h->basis_order() == -1)
  {
    warning("No data location in destination to interpolate to.");
    return;
  }

  FieldIPort *src_port = (FieldIPort *)get_iport("Source");
  FieldHandle fsrc_h;
  if (!(src_port->get(fsrc_h) && fsrc_h.get_rep()))
  {
    return;
  }
  if (fsrc_h->basis_order() == -1)
  {
    warning("No data location in Source field to interpolate from.");
    return;
  }
  
  CompileInfoHandle ci =
    BuildMappingMatrixAlgo::get_compile_info(fsrc_h->mesh()->get_type_description(),
					    fsrc_h->order_type_description(),
					    fdst_h->mesh()->get_type_description(),
					    fdst_h->order_type_description(),
					    fdst_h->get_type_description());
  Handle<BuildMappingMatrixAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  fsrc_h->mesh()->synchronize(Mesh::LOCATE_E);
  const int interp_basis = (interpolation_basis_.get() == "linear")?1:0;
  MatrixOPort *omp = (MatrixOPort *)get_oport("Mapping");
  omp->send(algo->execute(fsrc_h->mesh(), fdst_h->mesh(), interp_basis,
			  map_source_to_single_dest_.get(),
			  exhaustive_search_.get(),
			  exhaustive_search_max_dist_.get(), np_.get()));
}

CompileInfoHandle
BuildMappingMatrixAlgo::get_compile_info(const TypeDescription *msrc,
					const TypeDescription *lsrc,
					const TypeDescription *mdst,
					const TypeDescription *ldst,
					const TypeDescription *fdst)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("BuildMappingMatrixAlgoT");
  static const string base_class_name("BuildMappingMatrixAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       lsrc->get_filename() + "." +
		       ldst->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name() + ", " +
                       lsrc->get_name() + ", " +
                       mdst->get_name() + ", " +
                       ldst->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  fdst->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
