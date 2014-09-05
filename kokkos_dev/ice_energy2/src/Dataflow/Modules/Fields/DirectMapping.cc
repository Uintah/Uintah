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
#include <Dataflow/Network/Ports/FieldPort.h>
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
  GuiString  gui_interpolation_basis_;
  GuiInt     gui_map_source_to_single_dest_;
  GuiInt     gui_exhaustive_search_;
  GuiDouble  gui_exhaustive_search_max_dist_;
  GuiInt     gui_npts_;

  FieldHandle field_output_handle_;
};

DECLARE_MAKER(DirectMapping)
DirectMapping::DirectMapping(GuiContext* ctx) : 
  Module("DirectMapping", ctx, Filter, "FieldsData", "SCIRun"),
  gui_interpolation_basis_(get_ctx()->subVar("interpolation_basis"), "linear"),
  gui_map_source_to_single_dest_(get_ctx()->subVar("map_source_to_single_dest"), 0),
  gui_exhaustive_search_(get_ctx()->subVar("exhaustive_search"), 0),
  gui_exhaustive_search_max_dist_(get_ctx()->subVar("exhaustive_search_max_dist"), -1),
  gui_npts_(get_ctx()->subVar("np"), 1),
  field_output_handle_(0)
{
}

DirectMapping::~DirectMapping()
{
}

void
DirectMapping::execute()
{
  FieldHandle field_src_handle;
  FieldHandle field_dst_handle;

  if( !get_input_handle( "Source",      field_src_handle, true ) ) return;
  if( !get_input_handle( "Destination", field_dst_handle, true ) ) return;

  if( inputs_changed_  ||
      
      !field_src_handle.get_rep() ||
      !field_dst_handle.get_rep() ||
      
      gui_interpolation_basis_.changed( true ) ||
      gui_map_source_to_single_dest_ .changed( true )  ||
      gui_exhaustive_search_.changed( true ) ||
      gui_exhaustive_search_max_dist_.changed( true ) ||
      gui_npts_.changed( true ) ) {

    update_state(Executing);

    TypeDescription::td_vec *tdv = 
      field_src_handle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
    const string outputDataType = (*tdv)[0]->get_name();
    const string oftn =
      field_dst_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      field_dst_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      field_dst_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(outputDataType, 
                                                          0, "<", " >, ") +
      field_dst_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(outputDataType,
                                                          0, "<", " >") + " >";

    CompileInfoHandle ci =
      DirectMappingAlgo::get_compile_info(field_src_handle->get_type_description(),
                                          field_src_handle->order_type_description(),
                                          field_dst_handle->get_type_description(),
                                          field_dst_handle->order_type_description(),
					  oftn);

    Handle<DirectMappingAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    field_src_handle->mesh()->synchronize(Mesh::LOCATE_E);

    field_output_handle_ =
      algo->execute(field_src_handle, field_dst_handle->mesh(),
		    field_dst_handle->basis_order(),
		    gui_interpolation_basis_.get(),
		    gui_map_source_to_single_dest_.get(),
		    gui_exhaustive_search_.get(),
		    gui_exhaustive_search_max_dist_.get(),
		    gui_npts_.get());

    // Copy the properties from source field.
    if (field_output_handle_.get_rep())
      field_output_handle_->copy_properties(field_src_handle.get_rep());
  }

  // Send the data downstream
  send_output_handle("Remapped Destination", field_output_handle_, true);
}

CompileInfoHandle
DirectMappingAlgo::get_compile_info(const TypeDescription *fsrc,
                                    const TypeDescription *lsrc,
                                    const TypeDescription *fdst,
                                    const TypeDescription *ldst,
				    const string &fout)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectMappingAlgoT");
  static const string base_class_name("DirectMappingAlgo");

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
