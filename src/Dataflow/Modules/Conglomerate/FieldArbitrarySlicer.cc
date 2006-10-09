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

//    File   : FieldArbitrarySlicer.h
//    Author : Michael Callahan &&
//             Allen Sanderson
//             SCI Institute
//             University of Utah
//    Date   : March 2006
//
//    Copyright (C) 2006 SCI Group


#include <Dataflow/Modules/Conglomerate/FieldArbitrarySlicer.h>

#include <Core/Algorithms/Fields/ApplyMappingMatrix.h>
#include <Dataflow/Modules/Fields/ClipByFunction.h>
#include <Dataflow/Modules/Fields/TransformData.h>
#include <Dataflow/Modules/Visualization/Isosurface.h>

#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Containers/HashTable.h>

#include <Core/Basis/HexTrilinearLgn.h>

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif

namespace SCIRun {

SCISHARE MatrixHandle append_sparse_matrices(vector<MatrixHandle> &matrices);

class FieldArbitrarySlicer : public Module {
public:
  FieldArbitrarySlicer(GuiContext *context);

  virtual ~FieldArbitrarySlicer();

  virtual void execute();

private:
  FieldHandle field_transformed_handle_;
  FieldHandle field_output_handle_;

  //! GUI variables
  GuiString  gui_function_;
  GuiInt     gui_zero_checks_;

  GuiDouble  gui_slice_value_min_;
  GuiDouble  gui_slice_value_max_;
  GuiDouble  gui_slice_value_;
  GuiDouble  gui_slice_value_typed_;
  GuiInt     gui_slice_value_quantity_;
  GuiString  gui_slice_quantity_range_;
  GuiString  gui_slice_quantity_clusive_;
  GuiDouble  gui_slice_quantity_min_;
  GuiDouble  gui_slice_quantity_max_;
  GuiString  gui_slice_quantity_list_;
  GuiString  gui_slice_value_list_;
  GuiString  gui_slice_matrix_list_;
  GuiInt     gui_extract_from_new_field_;
  GuiInt     gui_use_algorithm_;
  GuiInt     gui_build_field_;
  GuiInt     gui_build_geom_;
  GuiString  gui_active_slice_value_selection_tab_;
  GuiString  gui_active_tab_;

  //gui_update_type_ must be declared after gui_iso_value_max_ which is
  //traced in the tcl code. If gui_update_type_ is set to Auto having it
  //last will prevent the net from executing when it is instantiated.
  GuiString  gui_update_type_;

  //! status variables
  vector< double > slicevals_;
 };


DECLARE_MAKER(FieldArbitrarySlicer)


FieldArbitrarySlicer::FieldArbitrarySlicer(GuiContext *context)
  : Module("FieldArbitrarySlicer", context, Filter, "FieldsCreate", "SCIRun"),
    
    field_transformed_handle_(0),
    field_output_handle_(0),

    gui_function_(get_ctx()->subVar("function"), "result = sqrt(x*x + y*y);"),

    gui_zero_checks_(context->subVar("zero-checks"), 1),

    gui_slice_value_min_(context->subVar("slice-value-min"),  0.0),
    gui_slice_value_max_(context->subVar("slice-value-max"), 99.0),
    gui_slice_value_(context->subVar("slice-value"), 0.0),
    gui_slice_value_typed_(context->subVar("slice-value-typed"), 0.0),
    gui_slice_value_quantity_(context->subVar("slice-value-quantity"), 1),
    gui_slice_quantity_range_(context->subVar("quantity-range"), "field"),
    gui_slice_quantity_clusive_(context->subVar("quantity-clusive"), "exclusive"),
    gui_slice_quantity_min_(context->subVar("quantity-min"),   0),
    gui_slice_quantity_max_(context->subVar("quantity-max"), 100),
    gui_slice_quantity_list_(context->subVar("quantity-list"), ""),
    gui_slice_value_list_(context->subVar("slice-value-list"), "No values present."),
    gui_slice_matrix_list_(context->subVar("matrix-list"),
			 "No matrix present - execution needed."),
    gui_extract_from_new_field_(context->subVar("extract-from-new-field"), 1),
    gui_use_algorithm_(context->subVar("algorithm"), 0),
    gui_build_field_(context->subVar("build_trisurf"), 1),
    gui_build_geom_(context->subVar("build_geom"), 1),
    gui_active_slice_value_selection_tab_(context->subVar("active-slice-value-selection-tab"), "0"),
    gui_active_tab_(context->subVar("active_tab"), "0"),
    gui_update_type_(context->subVar("update_type"), "On Release")
{
}


FieldArbitrarySlicer::~FieldArbitrarySlicer()
{
}


void
FieldArbitrarySlicer::execute()
{
  //! Get the input field handle from the port.
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input Field",  field_input_handle,  true  ) ) return;

  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before get.
  //////////////////////////////////////////////////////////////////////////////
  //! Transform Section
  if( gui_function_.changed( true ) )
    inputs_changed_ = true;

  //! Check to see if the input field has changed.
  if( inputs_changed_ ||
      !field_transformed_handle_.get_rep() ) {

    update_state(Executing);

    string function = gui_function_.get();

    //! Remove trailing white-space from the function string.
    while (function.size() && isspace(function[function.size()-1]))
      function.resize(function.size()-1);

    string outputDataType("double");

    const TypeDescription *ftd = field_input_handle->get_type_description();
    const TypeDescription *ltd = field_input_handle->order_type_description();
    const string oftn = 
      field_input_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      field_input_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      field_input_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(outputDataType, 
							 0, "<", " >, ") +
      field_input_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(outputDataType,
							 0, "<", " >") + " >";
    int hoffset = 0;
    Handle<TransformDataAlgo> algoTD;
    
    while (1) {
      CompileInfoHandle ci =
	TransformDataAlgo::get_compile_info(ftd, oftn, ltd, function, hoffset);
      if (!DynamicCompilation::compile(ci, algoTD, false, this)) {
	error("Your function would not compile.");
	get_gui()->eval(get_id() + " compile_error "+ci->filename_);
	DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	return;
      }

      if (algoTD->identify() == function) {
	break;
      }
      hoffset++;
    }

    field_transformed_handle_ = algoTD->execute(field_input_handle);
  }

  if( !field_transformed_handle_.get_rep() ) {
    error( "Can not find the transformed field for slicing" );
    return;
  } else {
    remark( "Transformed field for slicing is present" );
  }

  //////////////////////////////////////////////////////////////////////////////
  //! Isosurface Section

  // Check to see if the transformed field has changed.
  if( inputs_changed_ ) {

    update_state(Executing);

    ScalarFieldInterfaceHandle sfi =
      field_transformed_handle_->query_scalar_interface(this);

    if (!sfi.get_rep()) {
      error("Input field does not contain scalar data.");
      return;
    }

    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);

    // Check to see if the gui min max are different than the field.
    if( gui_slice_value_min_.get() != minmax.first ||
	gui_slice_value_max_.get() != minmax.second ) {

      gui_slice_value_min_.set( minmax.first );
      gui_slice_value_max_.set( minmax.second );

      ostringstream str;
      str << get_id() << " set_min_max ";
      get_gui()->execute(str.str().c_str());
    }

    if ( !gui_extract_from_new_field_.get() )
      return;
  }

  vector<double> slicevals(0);

  double qmin = gui_slice_value_min_.get();
  double qmax = gui_slice_value_max_.get();

  if (gui_active_slice_value_selection_tab_.get() == "0") { // slider / typed
    const double val = gui_slice_value_.get();
    const double valTyped = gui_slice_value_typed_.get();
    if (val != valTyped) {
      char s[1000];
      sprintf(s, "Typed slice value %g was out of range.  Using slice value %g instead.", valTyped, val);
      warning(s);
      gui_slice_value_typed_.set(val);
    }
    if ( qmin <= val && val <= qmax )
      slicevals.push_back(val);
    else {
      error("Typed slice value out of range -- skipping slice surfacing.");
      return;
    }
  } else if (gui_active_slice_value_selection_tab_.get() == "1") { // quantity
    int num = gui_slice_value_quantity_.get();

    if (num < 1) {
      error("Slice surface quantity must be at least one -- skipping slice surfacing.");
      return;
    }

    string range = gui_slice_quantity_range_.get();

    if (range == "manual") {
      qmin = gui_slice_quantity_min_.get();
      qmax = gui_slice_quantity_max_.get();
    } // else we're using "field" and qmax and qmin were set above
    
    if (qmin >= qmax) {
      error("Can't use quantity tab if the minimum and maximum are the same.");
      return;
    }

    string clusive = gui_slice_quantity_clusive_.get();

    ostringstream str;

    str << get_id() << " set-slice-quant-list \"";

    if (clusive == "exclusive") {
      // if the min - max range is 2 - 4, and the user requests 3 slicevals,
      // the code below generates 2.333, 3.0, and 3.666 -- which is nice
      // since it produces evenly spaced slices in torroidal data.
	
      double di=(qmax - qmin)/(double)num;
      for (int i=0; i<num; i++) {
	slicevals.push_back(qmin + ((double)i+0.5)*di);
	str << " " << slicevals[i];
      }
    } else if (clusive == "inclusive") {
      // if the min - max range is 2 - 4, and the user requests 3 slicevals,
      // the code below generates 2.0, 3.0, and 4.0.

      double di=(qmax - qmin)/(double)(num-1.0);
      for (int i=0; i<num; i++) {
	slicevals.push_back(qmin + ((double)i*di));
	str << " " << slicevals[i];
      }
    }

    str << "\"";

    get_gui()->execute(str.str().c_str());

  } else if (gui_active_slice_value_selection_tab_.get() == "2") { // list
    istringstream vlist(gui_slice_value_list_.get());
    double val;
    while(!vlist.eof()) {
      vlist >> val;
      if (vlist.fail()) {
	if (!vlist.eof()) {
	  vlist.clear();
	  warning("List of Slicevals was bad at character " +
		  to_string((int)(vlist.tellg())) +
		  "('" + ((char)(vlist.peek())) + "').");
	}
	break;
      }
      else if (!vlist.eof() && vlist.peek() == '%') {
	vlist.get();
	val = qmin + (qmax - qmin) * val / 100.0;
      }
      slicevals.push_back(val);
    }
  } else if (gui_active_slice_value_selection_tab_.get() == "3") { // matrix

    MatrixHandle matrix_input_handle;
    if( !get_input_handle( "Optional Slice values", matrix_input_handle, true ) )
      return;

    ostringstream str;
    
    str << get_id() << " set-slicematrix-list \"";

    for (int i=0; i < matrix_input_handle->nrows(); i++) {
      for (int j=0; j < matrix_input_handle->ncols(); j++) {
	slicevals.push_back(matrix_input_handle->get(i, j));

	str << " " << slicevals[i];
      }
    }

    str << "\"";

    get_gui()->execute(str.str().c_str());

  } else {
    error("Bad active_slice_value_selection_tab value");
    return;
  }

  // See if any of the slicevals have changed.
  if( slicevals_.size() != slicevals.size() ) {
    slicevals_.resize( slicevals.size() );
    inputs_changed_ = true;
  }

  for( unsigned int i=0; i<slicevals.size(); i++ ) {
    if( slicevals_[i] != slicevals[i] ) {
      slicevals_[i] = slicevals[i];
      inputs_changed_ = true;
    }
  }

  if( !field_output_handle_.get_rep() ||
      gui_zero_checks_.changed() ||
      inputs_changed_  ) {

    update_state(Executing);

    vector<FieldHandle> field_handles;
    vector<MatrixHandle> interpolant_handles;

    const TypeDescription *td =
      field_transformed_handle_->get_type_description();

    LockingHandle<MarchingCubesAlg> mc_alg;
    if (! mc_alg.get_rep()) {
      CompileInfoHandle ci = MarchingCubesAlg::get_compile_info(td);
      if (!module_dynamic_compile(ci, mc_alg)) {
	error( "Marching Cubes can not work with this field.");
	return;
      }
      mc_alg->set_np(1);
      mc_alg->set_field( field_transformed_handle_.get_rep() );

      for (unsigned int iv=0; iv<slicevals.size(); iv++)  {
	mc_alg->search( slicevals[iv],
			gui_build_field_.get(), gui_build_geom_.get() );
	field_handles.push_back( mc_alg->get_field(0) );
	interpolant_handles.push_back( mc_alg->get_interpolant(0) );
      }
      mc_alg->release();
    }

    FieldHandle   field_sliced_handle = 0;
    MatrixHandle matrix_sliced_handle = 0;

    //! Get the output field handle.
    if (field_handles.size() && field_handles[0].get_rep()) {

      //! Copy the name of field to the downstream field.
      string fldname;
      if (field_transformed_handle_->get_property("name", fldname)) {
	for (unsigned int i=0; i < field_handles.size(); i++)
	  field_handles[i]->set_property("name", fldname, false);
      } else {
	for (unsigned int i=0; i < field_handles.size(); i++)
	  field_handles[i]->set_property("name", string("Sliced Surface"), false);
      }

      //! No field.
      if (field_handles.size() == 0) {
	warning( "No slices found" );
	return;

      //! Single field.
      } else if (field_handles.size() == 1) {
	field_sliced_handle = field_handles[0];

      //! Multiple field_handles.
      } else {

	const TypeDescription *ftd = field_handles[0]->get_type_description();
	CompileInfoHandle ci = IsosurfaceAlgo::get_compile_info(ftd);
	
	Handle<IsosurfaceAlgo> algoIS;
	if (!module_dynamic_compile(ci, algoIS)) return;
	
	field_sliced_handle = algoIS->execute(field_handles);
      }

      //! Get the output interpolant handle.
      if (interpolant_handles[0].get_rep()) {
	if (interpolant_handles.size() == 1)
	  matrix_sliced_handle = interpolant_handles[0];
	else
	  matrix_sliced_handle = append_sparse_matrices(interpolant_handles);
      }
      else
	warning("Interpolant not computed for this input field type and data location.");
    }
    
    if( !field_sliced_handle.get_rep() ||
	!matrix_sliced_handle.get_rep() ) {
      error( "Can not find the sliced field and/or matrix for clipping" );
      return;
    } else {
      remark( "Sliced field and matrix for clipping is present" );
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Apply Mapping Section 1
    TypeDescription::td_vec *tdv1 = 
      field_input_handle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
    string accumtype1 = (*tdv1)[0]->get_name();
    if (field_input_handle->query_scalar_interface(this) != NULL) {
      accumtype1 = "double";
    }
    
    const string oftn1 = 
      field_sliced_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      field_sliced_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      field_sliced_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(accumtype1,
										     0, "<", " >, ") +
      field_sliced_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(accumtype1,
										     0, "<", " >") + " >";
    
    CompileInfoHandle ci1 =
      ApplyMappingMatrixAlgo::get_compile_info(field_input_handle->get_type_description(),
					       field_input_handle->order_type_description(),
					       field_sliced_handle->get_type_description(),
					       oftn1,
					       field_sliced_handle->order_type_description(),
					       field_input_handle->get_type_description(Field::FDATA_TD_E),
					       accumtype1);
    Handle<ApplyMappingMatrixAlgo> algoAMM1;
    if (!module_dynamic_compile(ci1, algoAMM1)) return;

    //! Apply the matrix to the sliced data.
    FieldHandle field_apply_mapping_handle =
      algoAMM1->execute(this,
			field_input_handle,
			field_sliced_handle->mesh(),
			matrix_sliced_handle);
    
    if( !field_apply_mapping_handle.get_rep() ) {
      error( "Can not find the first mapped field for the second mapping" );
      return;
    } else {
      remark( "First mapped field for the second mapping is present" );
    }

    if( !gui_zero_checks_.get () ) {
      field_output_handle_ = field_apply_mapping_handle;
    } else {
      //////////////////////////////////////////////////////////////////////////
      //! Clip By Function Section

      string function = gui_function_.get();

      string::size_type pos = function.find("result");
      if( pos != string::npos ) function.erase( pos, 6);

      pos = function.find("=");
      if( pos != string::npos ) function.erase( pos, 1);

      pos = function.find_last_of(";");
      if( pos != string::npos ) function.erase( pos, 1);

      //! remove trailing white-space from the function string
      while (function.size() && isspace(function[function.size()-1]))
	function.resize(function.size()-1);

      //! Add the test to the function
      function = string("fabs( (") + function + string(") - v) < 1e-2" );

      const TypeDescription *ftd = field_sliced_handle->get_type_description();
      Handle<ClipByFunctionAlgo> algoCBF;
      int hoffset = 0;

      while (1) {
	CompileInfoHandle ci =
	  ClipByFunctionAlgo::get_compile_info(ftd, function, hoffset);
	if (!DynamicCompilation::compile(ci, algoCBF, false, this)){
	  error("Your function would not compile.");
	  get_gui()->eval(get_id() + " compile_error "+ci->filename_);
	  DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	  return;
	}
	if (algoCBF->identify() == function)
	  break;

	hoffset++;
      }

      int gMode = -1;

      // User Variables.
      algoCBF->u0 = algoCBF->u1 = algoCBF->u2 = algoCBF->u3 =
	algoCBF->u4 = algoCBF->u5 = 0;

      if (!(field_sliced_handle->basis_order() == 0 && gMode == 0 ||
	    field_sliced_handle->basis_order() == 1 && gMode != 0) &&
	  field_sliced_handle->mesh()->dimensionality() > 0)
	{
	  warning("Basis doesn't match clip location, value will always be zero.");
	}

      MatrixHandle matrix_clipped_handle;

      FieldHandle field_clipped_handle =
	algoCBF->execute( this, field_sliced_handle,
			  gMode, matrix_clipped_handle );


    
      if( !field_clipped_handle.get_rep() ||
	  !matrix_clipped_handle.get_rep() ) {
	error( "Can not find the clipped field and/or matrix for the second mapping" );
	return;
      } else {
	remark( "Clipped field and matrix for the second mapping are present" );
      }

      //////////////////////////////////////////////////////////////////////////
      //! Apply Mapping Section 2
      TypeDescription::td_vec *tdv2 = 
	field_apply_mapping_handle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      string accumtype2 = (*tdv2)[0]->get_name();
      if (field_apply_mapping_handle->query_scalar_interface(this) != NULL) {
	accumtype2 = "double";
      }

      const string oftn2 = 
	field_clipped_handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
	field_clipped_handle->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
	field_clipped_handle->get_type_description(Field::BASIS_TD_E)->get_similar_name(accumtype2,
											0, "<", " >, ") +
	field_clipped_handle->get_type_description(Field::FDATA_TD_E)->get_similar_name(accumtype2,
											0, "<", " >") + " >";
       
      CompileInfoHandle ci2 =
	ApplyMappingMatrixAlgo::get_compile_info(field_apply_mapping_handle->get_type_description(),
						 field_apply_mapping_handle->order_type_description(),
						 field_clipped_handle->get_type_description(),
						 oftn2,
						 field_clipped_handle->order_type_description(),
						 field_apply_mapping_handle->get_type_description(Field::FDATA_TD_E),
						 accumtype2);
      Handle<ApplyMappingMatrixAlgo> algoAMM2;
      if (!module_dynamic_compile(ci2, algoAMM2)) return;

      //! Apply the matrix to the clipped data.
      field_output_handle_ =
	algoAMM2->execute(this,
			  field_apply_mapping_handle,
			  field_clipped_handle->mesh(),
			  matrix_clipped_handle);
    }
  }

  //! Send the data downstream.
  send_output_handle( "Output Field", field_output_handle_, true );
}


CompileInfoHandle
FieldArbitrarySlicerAlgo::get_compile_info(const TypeDescription *ftd,
				  const TypeDescription *ttd)
{
  //! use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldArbitrarySlicerAlgoT");
  static const string base_class_name("FieldArbitrarySlicerAlgo");

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
FieldArbitrarySlicerWorkAlgo::get_compile_info(const TypeDescription *iftd,
				      const TypeDescription *oftd,
				      bool geometry_irregular)
{
  //! use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldArbitrarySlicerWorkAlgoT");
  static const string base_class_name("FieldArbitrarySlicerWorkAlgo");

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
