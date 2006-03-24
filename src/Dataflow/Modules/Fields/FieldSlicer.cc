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
 *  FieldSlicer.cc:
 *
 *  Written by:
 *   Michael Callahan &
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Modules/Fields/FieldSlicer.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Core/Basis/HexTrilinearLgn.h>


namespace SCIRun {

class FieldSlicer : public Module {
public:
  FieldSlicer(GuiContext *context);

  virtual ~FieldSlicer();

  virtual void execute();

private:
  FieldHandle          field_out_handle_;
  MatrixHandle         matrix_out_handle_;

  GuiInt               gui_axis_;
  GuiInt               gui_dims_;

  GuiInt               gui_dim_i_;
  GuiInt               gui_dim_j_;
  GuiInt               gui_dim_k_;

  GuiInt               gui_index_i_;
  GuiInt               gui_index_j_;
  GuiInt               gui_index_k_;

  //! gui_update_type_ must be declared after all gui vars because
  //! some are traced in the tcl code. If updateType_ is set to Auto
  //! having it last will prevent the net from executing when it is
  //! instantiated.

  GuiString            gui_update_type_;
};


DECLARE_MAKER(FieldSlicer)


FieldSlicer::FieldSlicer(GuiContext *context)
  : Module("FieldSlicer", context, Filter, "FieldsCreate", "SCIRun"),
    
    field_out_handle_( 0 ),
    matrix_out_handle_( 0 ),

    gui_axis_(context->subVar("axis"), 2),
    gui_dims_(context->subVar("dims"), 3),

    gui_dim_i_(context->subVar("dim-i"), 1),
    gui_dim_j_(context->subVar("dim-j"), 1),
    gui_dim_k_(context->subVar("dim-k"), 1),

    gui_index_i_(context->subVar("index-i"), 1),
    gui_index_j_(context->subVar("index-j"), 1),
    gui_index_k_(context->subVar("index-k"), 1),

    gui_update_type_(context->subVar("update_type"), "Manual")
{
}


FieldSlicer::~FieldSlicer()
{
}


void
FieldSlicer::execute()
{
  FieldHandle field_in_handle;

  if( !get_input_handle( "Input Field",  field_in_handle,  true  ) ) return;
  if( !get_input_handle( "Input Matrix", matrix_out_handle_, false ) ) return;

  if( !(field_in_handle->mesh()->topology_geometry() & Mesh::STRUCTURED) ) {

    error( field_in_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Only availible for topologically structured data." );
    return;
  }

  if( field_in_handle->basis_order() != 1 ) {
    error( field_in_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Currently only available for node data." );
    return;
  }

  //! The matrix is optional.
  if( matrix_out_handle_.get_rep() &&
      (matrix_out_handle_->nrows() != 3 || matrix_out_handle_->ncols() != 3) ) {

    error( "Input matrix is not a 3x3 matrix" );
    return;
  }

  // Get the type and dimensions of the mesh.
  vector<unsigned int> dims;

  field_in_handle->mesh()->get_dim( dims );

  bool update_dims = false;

  if( gui_dims_.get() != (int) dims.size() ) {
    gui_dims_.set( dims.size() );
    update_dims = true;
  }

  if( dims.size() >= 1 ) {
    if( gui_dim_i_.get() != (int) dims[0] ) {
      gui_dim_i_.set( dims[1] );
      update_dims = true;
    }
  }

  if( dims.size() >= 2 ) {
    if( gui_dim_j_.get() != (int) dims[1] ) {
      gui_dim_j_.set( dims[1] );
      update_dims = true;
    }
  }

  if( dims.size() >= 3 ) {
    if( gui_dim_k_.get() != (int) dims[2] ) {
      gui_dim_k_.set( dims[2] );
      update_dims = true;
    }
  }

  //! Check to see if the gui dimensions are different than the field.
  if( update_dims ) {

    //! Dims has callback on it, so it must be set it after i, j, and k.
    ostringstream str;
    str << get_id() << " set_size ";
    get_gui()->execute(str.str().c_str());

    reset_vars();
  }

  //! An input matrix is present so use the values in it to override
  //! the variables set in the gui.
  if( matrix_out_handle_.get_rep() ) {

    if( gui_dim_i_.get() != matrix_out_handle_->get(0, 2) ||
	gui_dim_j_.get() != matrix_out_handle_->get(1, 2) ||
	gui_dim_k_.get() != matrix_out_handle_->get(2, 2) ) {
      ostringstream str;
      str << "The dimensions of the matrix slicing do match the field. "
	  << " Expected "
	  << gui_dim_i_.get() << " "
	  << gui_dim_j_.get() << " "
	  << gui_dim_k_.get()
	  << " Got "
	  << matrix_out_handle_->get(0, 2) << " "
	  << matrix_out_handle_->get(1, 2) << " "
	  << matrix_out_handle_->get(2, 2);
      
      error( str.str() );
      return;
    }

    //! Check to see what axis has been selected.
    for (int i=0; i < matrix_out_handle_->nrows(); i++) {
      if( matrix_out_handle_->get(i, 0) == 1 && gui_axis_.get() != i ) {
	gui_axis_.set( i );	
	inputs_changed_ = true;
      }
    }

    //! Check to see what index has been selected.
    if( gui_index_i_.get() != matrix_out_handle_->get(0, 1) ||
	gui_index_j_.get() != matrix_out_handle_->get(1, 1) ||
	gui_index_k_.get() != matrix_out_handle_->get(2, 1) ) {

      gui_index_i_.set( (int) matrix_out_handle_->get(0, 1) );
      gui_index_j_.set( (int) matrix_out_handle_->get(1, 1) );
      gui_index_k_.set( (int) matrix_out_handle_->get(2, 1) );

      ostringstream str;
      str << get_id() << " update_index ";
      
      get_gui()->execute(str.str().c_str());
      
      reset_vars();

      inputs_changed_ = true;
    }
  }

  //! If no data or ainput change recreate the mesh.
  if( inputs_changed_  ||
      
      !field_out_handle_.get_rep() ||
      !matrix_out_handle_.get_rep() ||
      
      gui_axis_.changed( true ) ||
      (gui_axis_.get() == 0 && gui_index_i_.changed( true )) ||
      (gui_axis_.get() == 1 && gui_index_j_.changed( true )) ||
      (gui_axis_.get() == 2 && gui_index_k_.changed( true )) ) {

    update_state(Executing);

    const TypeDescription *ftd = field_in_handle->get_type_description();
    const TypeDescription *ttd =
      field_in_handle->get_type_description(Field::FDATA_TD_E);

    CompileInfoHandle ci = FieldSlicerAlgo::get_compile_info(ftd,ttd);
    Handle<FieldSlicerAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    unsigned int index;
    if (gui_axis_.get() == 0) {
      index = gui_index_i_.get();
    } else if (gui_axis_.get() == 1) {
      index = gui_index_j_.get();
    } else {
      index = gui_index_k_.get();
    }

    field_out_handle_ = algo->execute(field_in_handle, gui_axis_.get());

    //! Now the new field is defined so do the work on it.
    const TypeDescription *iftd = field_in_handle->get_type_description();
    const TypeDescription *oftd = field_out_handle_->get_type_description();
    const bool geom_irreg =
      (field_in_handle->mesh()->topology_geometry() & Mesh::IRREGULAR);

    ci = FieldSlicerWorkAlgo::get_compile_info(iftd,oftd,geom_irreg);
    Handle<FieldSlicerWorkAlgo> workalgo;

    if (!module_dynamic_compile(ci, workalgo)) return;
  
    workalgo->execute( field_in_handle, field_out_handle_,
		       index, gui_axis_.get() );
  }

  //! Create the output matrix with the axis and index
  if( matrix_out_handle_ == 0 ) {
    DenseMatrix *selected = scinew DenseMatrix(3,3);

    for (int i=0; i < 3; i++)
      selected->put(i, 0, (double) (gui_axis_.get() == i) );

    selected->put(0, 1, gui_index_i_.get() );
    selected->put(1, 1, gui_index_j_.get() );
    selected->put(2, 1, gui_index_k_.get() );

    selected->put(0, 2, gui_dim_i_.get() );
    selected->put(1, 2, gui_dim_j_.get() );
    selected->put(2, 2, gui_dim_k_.get() );

    matrix_out_handle_ = MatrixHandle(selected);
  }

  // Send the data downstream
  send_output_handle( "Output Field",  field_out_handle_, true );
  send_output_handle( "Output Matrix", matrix_out_handle_, true );
}


CompileInfoHandle
FieldSlicerAlgo::get_compile_info(const TypeDescription *ftd,
				  const TypeDescription *ttd)
{
  //! use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldSlicerAlgoT");
  static const string base_class_name("FieldSlicerAlgo");

  TypeDescription::td_vec *tdv = ttd->get_sub_type();
  string odat = (*tdv)[0]->get_name();
  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " +
		       odat );

  //! Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/QuadBilinearLgn.h");
  rval->add_basis_include("../src/Core/Basis/CrvLinearLgn.h");
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  rval->add_basis_include("../src/Core/Basis/NoData.h");

  rval->add_mesh_include("../src/Core/Datatypes/StructCurveMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/StructQuadSurfMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/PointCloudMesh.h");

  ftd->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
FieldSlicerWorkAlgo::get_compile_info(const TypeDescription *iftd,
				      const TypeDescription *oftd,
				      bool geometry_irregular)
{
  //! use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldSlicerWorkAlgoT");
  static const string base_class_name("FieldSlicerWorkAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       iftd->get_filename() + "." +
		       oftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       iftd->get_name() + ", " +
		       oftd->get_name() );

  //! Add in the include path to compile this obj
  if(geometry_irregular)
    rval->add_pre_include( "#define SET_POINT_DEFINED 1");

  rval->add_include(include_path);  

  iftd->fill_compile_info(rval);
  return rval;
}

} //! End namespace SCIRun
