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
 *  DirectMapping.cc:  Build an interpolant field -- a field that says
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
#include <Dataflow/Modules/Fields/DirectMapping.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class DirectMapping : public Module
{
public:
  DirectMapping(GuiContext* ctx);
  virtual ~DirectMapping();
  virtual void execute();

private:
  GuiString  gInterpolation_basis_;
  GuiInt     gMap_source_to_single_dest_;
  GuiInt     gExhaustive_search_;
  GuiDouble  gExhaustive_search_max_dist_;
  GuiInt     gNp_;

  std::string  interpolation_basis_;
  int     map_source_to_single_dest_;
  int     exhaustive_search_;
  double  exhaustive_search_max_dist_;
  int     np_;

  FieldHandle fHandle_;

  int sfGeneration_;
  int dfGeneration_;

  bool error_;
};

DECLARE_MAKER(DirectMapping)
DirectMapping::DirectMapping(GuiContext* ctx) : 
  Module("DirectMapping", ctx, Filter, "FieldsData", "SCIRun"),
  gInterpolation_basis_(ctx->subVar("interpolation_basis")),
  gMap_source_to_single_dest_(ctx->subVar("map_source_to_single_dest")),
  gExhaustive_search_(ctx->subVar("exhaustive_search")),
  gExhaustive_search_max_dist_(ctx->subVar("exhaustive_search_max_dist")),
  gNp_(ctx->subVar("np")),
  sfGeneration_(-1),
  dfGeneration_(-1),
  error_(false)
{
}

DirectMapping::~DirectMapping()
{
}

void
DirectMapping::execute()
{
  update_state(NeedData);

  FieldIPort * sfield_port = (FieldIPort *)get_iport("Source");
  FieldHandle sfHandle;
  if (!(sfield_port->get(sfHandle) && sfHandle.get_rep())) {
    error( "No source field handle or representation" );
    return;
  }
  if (sfHandle->basis_order() == -1) {
    error("No data basis in source field to interpolate from.");
    return;
  }

  FieldIPort *dfield_port = (FieldIPort *)get_iport("Destination");
  FieldHandle dfHandle;
  if (!(dfield_port->get(dfHandle) && dfHandle.get_rep())) {
    error( "No destination field handle or representation" );
    return;
  }
  if (dfHandle->basis_order() == -1) {
    error("No data basis in destination field to interpolate to.");
    return;
  }

  bool update = false;

  // Check to see if the source field has changed.
  if( sfGeneration_ != sfHandle->generation ) {
    sfGeneration_ = sfHandle->generation;
    update = true;
  }

  // Check to see if the destination field has changed.
  if( dfGeneration_ != dfHandle->generation ) {
    dfGeneration_ = dfHandle->generation;
    update = true;
  }

  std::string interpolation_basis = gInterpolation_basis_.get();
  int map_source_to_single_dest = gMap_source_to_single_dest_.get();
  int exhaustive_search = gExhaustive_search_.get();
  double exhaustive_search_max_dist = gExhaustive_search_max_dist_.get();
  int np = gNp_.get();
  
  if( interpolation_basis_ != interpolation_basis ||
      map_source_to_single_dest_  != map_source_to_single_dest  ||
      exhaustive_search_ != exhaustive_search ||
      exhaustive_search_max_dist_     != exhaustive_search_max_dist ||
      np_        != np ) {

    interpolation_basis_ = interpolation_basis;
    map_source_to_single_dest_  = map_source_to_single_dest;
    exhaustive_search_ = exhaustive_search;
    exhaustive_search_max_dist_     = exhaustive_search_max_dist;
    np_        = np;

    update = true;
  }

  if( !fHandle_.get_rep() ||
      update ||
      error_ ) {

    error_ = false;

    CompileInfoHandle ci =
      DirectMappingAlgo::get_compile_info(sfHandle->get_type_description(),
                                          sfHandle->order_type_description(),
                                          dfHandle->get_type_description(),
                                          dfHandle->order_type_description());

    Handle<DirectMappingAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    sfHandle->mesh()->synchronize(Mesh::LOCATE_E);

    fHandle_ = algo->execute(sfHandle, dfHandle->mesh(),
			     dfHandle->basis_order(),
			     interpolation_basis_,
			     map_source_to_single_dest_,
			     exhaustive_search_,
			     exhaustive_search_max_dist_, np_);

    // Copy the properties from source field.
    if (fHandle_.get_rep())
      fHandle_->copy_properties(sfHandle.get_rep());
  }

  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Remapped Destination");
    ofield_port->send( fHandle_ );
  }
}

CompileInfoHandle
DirectMappingAlgo::get_compile_info(const TypeDescription *fsrc,
                                    const TypeDescription *lsrc,
                                    const TypeDescription *fdst,
                                    const TypeDescription *ldst)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectMappingAlgoT");
  static const string base_class_name("DirectMappingAlgo");

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
