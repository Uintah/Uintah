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


//!    File   : FieldSlicer.h
//!    Author : Michael Callahan &&
//!             Allen Sanderson
//!             SCI Institute
//!             University of Utah
//!    Date   : March 2006
//
//!    Copyright (C) 2006 SCI Group


#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Modules/Fields/FieldSubSample.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/CrvLinearLgn.h>

namespace SCIRun {

class FieldSubSample : public Module {
public:
  FieldSubSample(GuiContext *context);

  virtual ~FieldSubSample();

  virtual void execute();

private:
  FieldHandle  field_out_handle_;
  MatrixHandle matrix_out_handle_;

  GuiInt gui_power_app_;

  GuiInt gui_wrap_;
  GuiInt gui_dims_;

  GuiInt gui_dim_i_;
  GuiInt gui_dim_j_;
  GuiInt gui_dim_k_;

  GuiInt gui_start_i_;
  GuiInt gui_start_j_;
  GuiInt gui_start_k_;

  GuiInt gui_stop_i_;
  GuiInt gui_stop_j_;
  GuiInt gui_stop_k_;

  GuiInt gui_stride_i_;
  GuiInt gui_stride_j_;
  GuiInt gui_stride_k_;

  GuiInt gui_wrap_i_;
  GuiInt gui_wrap_j_;
  GuiInt gui_wrap_k_;
};


DECLARE_MAKER(FieldSubSample)


FieldSubSample::FieldSubSample(GuiContext *context)
  : Module("FieldSubSample", context, Filter, "FieldsCreate", "SCIRun"),
    field_out_handle_( 0 ),
    matrix_out_handle_( 0 ),

    gui_power_app_(context->subVar("power_app"), 0),

    gui_wrap_(context->subVar("wrap"), 0 ),
    gui_dims_(context->subVar("dims"), 3 ),

    gui_dim_i_(context->subVar("dim-i"), 2),
    gui_dim_j_(context->subVar("dim-j"), 2),
    gui_dim_k_(context->subVar("dim-k"), 2),

    gui_start_i_(context->subVar("start-i"), 0),
    gui_start_j_(context->subVar("start-j"), 0),
    gui_start_k_(context->subVar("start-k"), 0),

    gui_stop_i_(context->subVar("stop-i"), 1),
    gui_stop_j_(context->subVar("stop-j"), 1),
    gui_stop_k_(context->subVar("stop-k"), 1),

    gui_stride_i_(context->subVar("stride-i"), 1),
    gui_stride_j_(context->subVar("stride-j"), 1),
    gui_stride_k_(context->subVar("stride-k"), 1),
 
    gui_wrap_i_(context->subVar("wrap-i"), 0),
    gui_wrap_j_(context->subVar("wrap-j"), 0),
    gui_wrap_k_(context->subVar("wrap-k"), 0)
{
}


FieldSubSample::~FieldSubSample()
{
}


void
FieldSubSample::execute()
{
  FieldHandle field_in_handle;

  //! Get the input field handle from the port.
  if( !get_input_handle( "Input Field",  field_in_handle,  true  ) ) return;

  //! Get the optional matrix handle from the port. Note if a matrix is
  //! present it is sent down stream. Otherwise it will be created.
  get_input_handle( "Input Matrix", matrix_out_handle_, false );

  // Because the field slicer is index based it can only work on
  // structured data. For unstructured data SamplePlane should be used.
  if( !(field_in_handle->mesh()->topology_geometry() & Mesh::STRUCTURED) ) {

    error( field_in_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Only availible for topologically structured data." );
    error( "For topologically unstructured data use SamplePlane." );
    return;
  }

  //! For now ?? Slice only node and cell based data.
  if( field_in_handle->basis_order() != 0 &&
      field_in_handle->basis_order() != 1 ) {
    error( field_in_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Currently only available for cell or node data." );
    return;
  }

  // Get the dimensions of the mesh.
  vector<unsigned int> dims;

  field_in_handle->mesh()->get_dim( dims );

  bool update_dims = false;

  //! Structured data with irregular points can be wrapped 
  int wrap = (field_in_handle->mesh()->topology_geometry() & Mesh::IRREGULAR);

  //! Check to see if the gui wrap is different than the field.
  if( gui_wrap_.get() != wrap ) {
    gui_wrap_.set( wrap );
    update_dims = true;
  }

  if( dims.size() >= 1 ) {
    //! Check to see if the gui dimensions are different than the field.
    if( gui_dim_i_.get() != (int) dims[0] ) {
      gui_dim_i_.set( dims[1] );
      update_dims = true;
    }
  }

  if( dims.size() >= 2 ) {
    //! Check to see if the gui dimensions are different than the field.
    if( gui_dim_j_.get() != (int) dims[1] ) {
      gui_dim_j_.set( dims[1] );
      update_dims = true;
    }
  }

  if( dims.size() >= 3 ) {
    //! Check to see if the gui dimensions are different than the field.
    if( gui_dim_k_.get() != (int) dims[2] ) {
      gui_dim_k_.set( dims[2] );
      update_dims = true;
    }
  }

  //! Check to see if the gui dimensions are different than the field.
  //! This is last because the GUI var has a callback on it.
  if( gui_dims_.get() != (int) dims.size() ) {
    gui_dims_.set( dims.size() );
    update_dims = true;
  }

  //! If the gui dimensions are different than the field then update the gui.
  if( update_dims ) {

    ostringstream str;
    str << get_id() << " set_size ";
    get_gui()->execute(str.str().c_str());

    reset_vars();
  }

  //! An input matrix is present so use the values in it to override
  //! the variables set in the gui.
  //! Column 0 start  index to subsample.
  //! Column 1 stop   index to subsample.
  //! Column 2 stride value to subsample.
  //! Column 3 wrap flag.
  //! Column 4 dimensions of the data.
  if( matrix_out_handle_.get_rep() ) {

    //! The matrix is optional. If present make sure it is a 3x5 matrix.
    //! The row indices is the axis index. The column is the data.
    if( (matrix_out_handle_->nrows() != 3 ||
	 matrix_out_handle_->ncols() != 5) ) {
      
      error( "Input matrix is not a 3x5 matrix" );
      return;
    }

    //! Sanity check. Make sure the gui dimensions match the matrix
    //! dimensions.
    if( gui_dim_i_.get() != matrix_out_handle_->get(0, 4) ||
	gui_dim_j_.get() != matrix_out_handle_->get(1, 4) ||
	gui_dim_k_.get() != matrix_out_handle_->get(2, 4) ) {
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

    //! Check to see what index has been selected and if it matches
    //! the gui index.
    if( gui_start_i_.get() != (int) matrix_out_handle_->get(0, 0) ||
	gui_start_j_.get() != (int) matrix_out_handle_->get(1, 0) ||
	gui_start_k_.get() != (int) matrix_out_handle_->get(2, 0) ||

	gui_stop_i_.get() != (int) matrix_out_handle_->get(0, 1) ||
	gui_stop_j_.get() != (int) matrix_out_handle_->get(1, 1) ||
	gui_stop_k_.get() != (int) matrix_out_handle_->get(2, 1) ||

	gui_stride_i_.get() != (int) matrix_out_handle_->get(0, 2) ||
	gui_stride_j_.get() != (int) matrix_out_handle_->get(1, 2) ||
	gui_stride_k_.get() != (int) matrix_out_handle_->get(2, 2) ||

	gui_wrap_i_.get() != (int) matrix_out_handle_->get(0, 3) ||
	gui_wrap_j_.get() != (int) matrix_out_handle_->get(1, 3) ||
	gui_wrap_k_.get() != (int) matrix_out_handle_->get(2, 3) ) {

      gui_start_i_.set( (int) matrix_out_handle_->get(0, 0) );
      gui_start_j_.set( (int) matrix_out_handle_->get(1, 0) );
      gui_start_k_.set( (int) matrix_out_handle_->get(2, 0) );

      gui_stop_i_.set( (int) matrix_out_handle_->get(0, 1) );
      gui_stop_j_.set( (int) matrix_out_handle_->get(1, 1) );
      gui_stop_k_.set( (int) matrix_out_handle_->get(2, 1) );

      gui_stride_i_.set( (int) matrix_out_handle_->get(0, 2) );
      gui_stride_j_.set( (int) matrix_out_handle_->get(1, 2) );
      gui_stride_k_.set( (int) matrix_out_handle_->get(2, 2) );

      gui_wrap_i_.set( (int) matrix_out_handle_->get(0, 3) );
      gui_wrap_j_.set( (int) matrix_out_handle_->get(1, 3) );
      gui_wrap_k_.set( (int) matrix_out_handle_->get(2, 3) );

      ostringstream str;
      str << get_id() << " update_index ";

      get_gui()->execute(str.str().c_str());
	
      reset_vars();

      inputs_changed_ = true;
    }
  }

  //! If no data or an input change recreate the field. I.e Only
  //! execute when neeed.
  if( inputs_changed_ ||

      !field_out_handle_.get_rep() ||
      !matrix_out_handle_.get_rep() ||

      gui_start_i_.changed( true ) ||
      gui_start_j_.changed( true ) ||
      gui_start_k_.changed( true ) ||
      
      gui_stop_i_.changed( true ) ||
      gui_stop_j_.changed( true ) ||
      gui_stop_k_.changed( true ) ||
      
      gui_stride_i_.changed( true ) ||
      gui_stride_j_.changed( true ) ||
      gui_stride_k_.changed( true ) ||
      
      gui_wrap_i_.changed( true ) ||
      gui_wrap_j_.changed( true ) ||
      gui_wrap_k_.changed( true ) ) {

    // Update the state. Other state changes are handled in either
    // getting handles or in the calling method Module::do_execute.
    update_state(Executing);

    //! Create the new field via a dynamically compiled algorithm.
    const TypeDescription *ftd = field_in_handle->get_type_description();
    const bool geom_irreg =
      (field_in_handle->mesh()->topology_geometry() & Mesh::IRREGULAR);

    CompileInfoHandle ci =
      FieldSubSampleAlgo::get_compile_info(ftd,geom_irreg);

    Handle<FieldSubSampleAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    field_out_handle_ = algo->execute(field_in_handle,
	     gui_start_i_.get(),  gui_start_j_.get(),  gui_start_k_.get(),
	     gui_stop_i_.get(),   gui_stop_j_.get(),   gui_stop_k_.get(),
	     gui_stride_i_.get(), gui_stride_j_.get(), gui_stride_k_.get(),
	     gui_wrap_i_.get(),   gui_wrap_j_.get(),   gui_wrap_k_.get());

    //! Create the output matrix with the stop, stop, stride, wrap, and
    //! dimensions.
    if( matrix_out_handle_ == 0 ) {
      DenseMatrix *selected = scinew DenseMatrix(3,5);

      selected->put(0, 0, gui_start_i_.get() );
      selected->put(1, 0, gui_start_j_.get() );
      selected->put(2, 0, gui_start_k_.get() );

      selected->put(0, 1, gui_stop_i_.get() );
      selected->put(1, 1, gui_stop_j_.get() );
      selected->put(2, 1, gui_stop_k_.get() );

      selected->put(0, 2, gui_stride_i_.get() );
      selected->put(1, 2, gui_stride_j_.get() );
      selected->put(2, 2, gui_stride_k_.get() );

      selected->put(0, 3, gui_wrap_i_.get() );
      selected->put(1, 3, gui_wrap_j_.get() );
      selected->put(2, 3, gui_wrap_k_.get() );

      selected->put(0, 4, gui_dim_i_.get() );
      selected->put(1, 4, gui_dim_j_.get() );
      selected->put(2, 4, gui_dim_k_.get() );

      matrix_out_handle_ = MatrixHandle(selected);
    }
  }

  // Send the data downstream
  send_output_handle( "Output Field",  field_out_handle_, true );
  send_output_handle( "Output Matrix", matrix_out_handle_, true );
}

CompileInfoHandle
FieldSubSampleAlgo::get_compile_info(const TypeDescription *ftd,
				     bool geometry_irregular)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldSubSampleAlgoT");
  static const string base_class_name("FieldSubSampleAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  if(geometry_irregular)
    rval->add_pre_include( "#define SET_POINT_DEFINED 1");

  rval->add_include(include_path);

  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
