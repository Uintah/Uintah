/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

//    File   : ExtractIsosurface.cc
//    Author : Yarden Livnat
//    Date   : Fri Jun 15 16:38:02 2001


#include <Core/Algorithms/Visualization/ExtractIsosurface.h>

#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>

#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>

#ifdef _WIN32
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE
#endif

namespace SCIRun {

class ExtractIsosurface : public Module {

public:
  ExtractIsosurface(GuiContext* ctx);
  virtual ~ExtractIsosurface();
  virtual void execute();

private:
  FieldHandle  field_output_handle_;
  MatrixHandle matrix_output_handle_;
  GeomHandle   geometry_output_handle_;

  //! GUI variables
  GuiDouble  gui_iso_value_min_;
  GuiDouble  gui_iso_value_max_;
  GuiDouble  gui_iso_value_;
  GuiDouble  gui_iso_value_typed_;
  GuiInt     gui_iso_value_quantity_;
  GuiString  gui_iso_quantity_range_;
  GuiString  gui_iso_quantity_clusive_;
  GuiDouble  gui_iso_quantity_min_;
  GuiDouble  gui_iso_quantity_max_;
  GuiString  gui_iso_quantity_list_;
  GuiString  gui_iso_value_list_;
  GuiString  gui_iso_matrix_list_;
  GuiInt     gui_extract_from_new_field_;
  GuiInt     gui_use_algorithm_;
  GuiInt     gui_build_field_;
  GuiInt     gui_build_geom_;
  GuiInt     gui_np_;          
  GuiString  gui_active_isoval_selection_tab_;
  GuiString  gui_active_tab_; 
  //gui_update_type_ must be declared after gui_iso_value_max_ which is
  //traced in the tcl code. If gui_update_type_ is set to Auto having it
  //last will prevent the net from executing when it is instantiated.
  GuiString  gui_update_type_;

  GuiDouble  gui_color_r_;
  GuiDouble  gui_color_g_;
  GuiDouble  gui_color_b_;

  //! status variables
  vector< double > isovals_;
};

DECLARE_MAKER(ExtractIsosurface)


ExtractIsosurface::ExtractIsosurface(GuiContext* context) : 
  Module("ExtractIsosurface", context, Filter, "Visualization", "SCIRun"), 

  field_output_handle_(0),
  matrix_output_handle_(0),
  geometry_output_handle_(0),

  gui_iso_value_min_(context->subVar("isoval-min"),  0.0),
  gui_iso_value_max_(context->subVar("isoval-max"), 99.0),
  gui_iso_value_(context->subVar("isoval"), 0.0),
  gui_iso_value_typed_(context->subVar("isoval-typed"), 0.0),
  gui_iso_value_quantity_(context->subVar("isoval-quantity"), 1),
  gui_iso_quantity_range_(context->subVar("quantity-range"), "field"),
  gui_iso_quantity_clusive_(context->subVar("quantity-clusive"), "exclusive"),
  gui_iso_quantity_min_(context->subVar("quantity-min"),   0),
  gui_iso_quantity_max_(context->subVar("quantity-max"), 100),
  gui_iso_quantity_list_(context->subVar("quantity-list"), ""),
  gui_iso_value_list_(context->subVar("isoval-list"), "No values present."),
  gui_iso_matrix_list_(context->subVar("matrix-list"),
		       "No matrix present - execution needed."),
  gui_extract_from_new_field_(context->subVar("extract-from-new-field"), 1),
  gui_use_algorithm_(context->subVar("algorithm"), 0),
  gui_build_field_(context->subVar("build_trisurf"), 1),
  gui_build_geom_(context->subVar("build_geom"), 1),
  gui_np_(context->subVar("np"), 1),
  gui_active_isoval_selection_tab_(context->subVar("active-isoval-selection-tab"),
				   "0"),
  gui_active_tab_(context->subVar("active_tab"), "0"),
  gui_update_type_(context->subVar("update_type"), "On Release"),
  gui_color_r_(context->subVar("color-r"), 0.4),
  gui_color_g_(context->subVar("color-g"), 0.2),
  gui_color_b_(context->subVar("color-b"), 0.9)
{
}


ExtractIsosurface::~ExtractIsosurface()
{
}


SCISHARE MatrixHandle
append_sparse_matrices(vector<MatrixHandle> &matrices)
{
  unsigned int i;
  int j;

  int ncols = matrices[0]->ncols();
  int nrows = 0;
  int nnz = 0;
  for (i = 0; i < matrices.size(); i++) {
    SparseRowMatrix *sparse = matrices[i]->sparse();
    nrows += sparse->nrows();
    nnz += sparse->nnz;
  }

  int *rr = scinew int[nrows+1];
  int *cc = scinew int[nnz];
  double *dd = scinew double[nnz];

  int offset = 0;
  int nnzcounter = 0;
  int rowcounter = 0;
  for (i = 0; i < matrices.size(); i++) {
    SparseRowMatrix *sparse = matrices[i]->sparse();
    for (j = 0; j < sparse->nnz; j++) {
      cc[nnzcounter] = sparse->columns[j];
      dd[nnzcounter] = sparse->a[j];
      nnzcounter++;
    }
    const int snrows = sparse->nrows();
    for (j = 0; j <= snrows; j++) {
      rr[rowcounter] = sparse->rows[j] + offset;
      rowcounter++;
    }
    rowcounter--;
    offset += sparse->rows[snrows];
  }

  return scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, dd);
}


void
ExtractIsosurface::execute()
{
  FieldHandle field_input_handle;
  if( !get_input_handle( "Field", field_input_handle, true ) ) return;

  // Check to see if the input field has changed.
  if( inputs_changed_ ) {

    ScalarFieldInterfaceHandle sfi =
      field_input_handle->query_scalar_interface(this);

    if (!sfi.get_rep()) {
      error("Input field does not contain scalar data.");
      return;
    }

    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);

    // Check to see if the gui min max are different than the field.
    if( gui_iso_value_min_.get() != minmax.first ||
	gui_iso_value_max_.get() != minmax.second ) {

      gui_iso_value_min_.set( minmax.first );
      gui_iso_value_max_.set( minmax.second );

      ostringstream str;
      str << get_id() << " set_min_max ";
      get_gui()->execute(str.str().c_str());
    }

    if ( !gui_extract_from_new_field_.get() )
      return;
  }

  // Get the optional colormap for the geometry.
  ColorMapHandle colormap_input_handle = 0;
  get_input_handle( "Optional Color Map", colormap_input_handle, false );
  
  vector<double> isovals(0);

  double qmin = gui_iso_value_min_.get();
  double qmax = gui_iso_value_max_.get();

  if (gui_active_isoval_selection_tab_.get() == "0") { // slider / typed
    const double val = gui_iso_value_.get();
    const double valTyped = gui_iso_value_typed_.get();
    if (val != valTyped) {
      char s[1000];
      sprintf(s, "Typed isovalue %g was out of range.  Using isovalue %g instead.", valTyped, val);
      warning(s);
      gui_iso_value_typed_.set(val);
    }
    if ( qmin <= val && val <= qmax )
      isovals.push_back(val);
    else {
      error("Typed isovalue out of range -- skipping isosurfacing.");
      return;
    }
  } else if (gui_active_isoval_selection_tab_.get() == "1") { // quantity
    int num = gui_iso_value_quantity_.get();

    if (num < 1) {
      error("ExtractIsosurface quantity must be at least one -- skipping isosurfacing.");
      return;
    }

    string range = gui_iso_quantity_range_.get();

    if (range == "colormap") {
      if (colormap_input_handle.get_rep() ) {
	error("No color colormap for isovalue quantity");
	return;
      }
      qmin = colormap_input_handle->getMin();
      qmax = colormap_input_handle->getMax();
    } else if (range == "manual") {
      qmin = gui_iso_quantity_min_.get();
      qmax = gui_iso_quantity_max_.get();
    } // else we're using "field" and qmax and qmin were set above
    
    if (qmin >= qmax) {
      error("Can't use quantity tab if the minimum and maximum are the same.");
      return;
    }

    string clusive = gui_iso_quantity_clusive_.get();

    ostringstream str;

    str << get_id() << " set-isoquant-list \"";

    if (clusive == "exclusive") {
      // if the min - max range is 2 - 4, and the user requests 3 isovals,
      // the code below generates 2.333, 3.0, and 3.666 -- which is nice
      // since it produces evenly spaced slices in torroidal data.
	
      double di=(qmax - qmin)/(double)num;
      for (int i=0; i<num; i++) {
	isovals.push_back(qmin + ((double)i+0.5)*di);
	str << " " << isovals[i];
      }
    } else if (clusive == "inclusive") {
      // if the min - max range is 2 - 4, and the user requests 3 isovals,
      // the code below generates 2.0, 3.0, and 4.0.

      double di=(qmax - qmin)/(double)(num-1.0);
      for (int i=0; i<num; i++) {
	isovals.push_back(qmin + ((double)i*di));
	str << " " << isovals[i];
      }
    }

    str << "\"";

    get_gui()->execute(str.str().c_str());

  } else if (gui_active_isoval_selection_tab_.get() == "2") { // list
    istringstream vlist(gui_iso_value_list_.get());
    double val;
    while(!vlist.eof()) {
      vlist >> val;
      if (vlist.fail()) {
	if (!vlist.eof()) {
	  vlist.clear();
	  warning("List of Isovals was bad at character " +
		  to_string((int)(vlist.tellg())) +
		  "('" + ((char)(vlist.peek())) + "').");
	}
	break;
      }
      else if (!vlist.eof() && vlist.peek() == '%') {
	vlist.get();
	val = qmin + (qmax - qmin) * val / 100.0;
      }
      isovals.push_back(val);
    }
  } else if (gui_active_isoval_selection_tab_.get() == "3") { // matrix

    MatrixHandle matrix_input_handle;
    if( !get_input_handle( "Optional Isovalues", matrix_input_handle, true ) )
      return;

    ostringstream str;
    
    str << get_id() << " set-isomatrix-list \"";

    for (int i=0; i < matrix_input_handle->nrows(); i++) {
      for (int j=0; j < matrix_input_handle->ncols(); j++) {
	isovals.push_back(matrix_input_handle->get(i, j));

	str << " " << isovals[i];
      }
    }

    str << "\"";

    get_gui()->execute(str.str().c_str());

  } else {
    error("Bad active_isoval_selection_tab value");
    return;
  }

  // See if any of the isovalues have changed.
  if( isovals_.size() != isovals.size() ) {
    isovals_.resize( isovals.size() );
    inputs_changed_ = true;
  }

  for( unsigned int i=0; i<isovals.size(); i++ ) {
    if( isovals_[i] != isovals[i] ) {
      isovals_[i] = isovals[i];
      inputs_changed_ = true;
    }
  }

  if( gui_use_algorithm_.changed( true ) ||
      gui_build_field_.changed( true ) ||
      gui_build_geom_.changed( true ) ||

      gui_np_.changed( true ) ||

      gui_color_r_.changed( true ) ||
      gui_color_g_.changed( true ) ||
      gui_color_b_.changed( true ) ) {

    inputs_changed_ = true;
  }

  // Decide if an interpolant will be computed for the output field.
  MatrixOPort *omatrix_port = (MatrixOPort *) get_oport("Mapping");

  const bool build_interp =
    gui_build_field_.get() && omatrix_port->nconnections();

  if( (gui_build_field_.get() && !field_output_handle_.get_rep()) ||
      (gui_build_geom_.get()  && !geometry_output_handle_.get_rep()) ||
      (build_interp           && !matrix_output_handle_.get_rep()) ||
      inputs_changed_  ) {

    update_state(Executing);

    vector<GeomHandle > geometry_handles;
    vector<FieldHandle> field_handles;
    vector<MatrixHandle> interpolant_handles;

    const TypeDescription *td = field_input_handle->get_type_description();

    switch (gui_use_algorithm_.get()) {
    case 0:  // Marching Cubes
      {
	LockingHandle<MarchingCubesAlg> mc_alg;
	if (! mc_alg.get_rep()) {
	  CompileInfoHandle ci = MarchingCubesAlg::get_compile_info(td);
	  if (!module_dynamic_compile(ci, mc_alg)) {
	    error( "Marching Cubes can not work with this field.");
	    return;
	  }
	  int np = gui_np_.get();
	  if (np <  1 ) { np =  1; gui_np_.set(np); }
	  if (np > 32 ) { np = 32; gui_np_.set(np); }
	  mc_alg->set_np(np);
	  mc_alg->set_field( field_input_handle.get_rep() );

	  for (unsigned int iv=0; iv<isovals.size(); iv++)  {
	    mc_alg->search( isovals[iv],
			    gui_build_field_.get(), gui_build_geom_.get() );
	    geometry_handles.push_back( mc_alg->get_geom() );
	    for (int i = 0 ; i < np; i++) {
	      field_handles.push_back( mc_alg->get_field(i) );
	      if (build_interp)
		interpolant_handles.push_back( mc_alg->get_interpolant(i) );
	    }
	  }
	  mc_alg->release();
	}
      }
      break;
    case 1:  // Noise
      {
	LockingHandle<NoiseAlg> noise_alg;
	if (! noise_alg.get_rep()) {
	  CompileInfoHandle ci =
	    NoiseAlg::get_compile_info(td,
				       field_input_handle->basis_order() == 0,
				       false);
	  if (! module_dynamic_compile(ci, noise_alg)) {
	    error( "NOISE can not work with this field.");
	    return;
	  }
	  noise_alg->set_field(field_input_handle.get_rep());

	  for (unsigned int iv=0; iv<isovals.size(); iv++) {
	    geometry_handles.push_back(noise_alg->search(isovals[iv],
			   gui_build_field_.get(), gui_build_geom_.get() ) );
	    field_handles.push_back(noise_alg->get_field());
	    if (build_interp)
	      interpolant_handles.push_back(noise_alg->get_interpolant());
	  }
	  noise_alg->release();
	}
      }
      break;

    case 2:  // View Dependent
      {
	LockingHandle<SageAlg> sage_alg;
	if (! sage_alg.get_rep()){
	  CompileInfoHandle ci = SageAlg::get_compile_info(td);
	  if (! module_dynamic_compile(ci, sage_alg)) {
	    error( "SAGE can not work with this field.");
	    return;
	  }
	  sage_alg->set_field(field_input_handle.get_rep());

	  for (unsigned int iv=0; iv<isovals.size(); iv++) {
	    GeomGroup *group = scinew GeomGroup;
	    GeomPoints *points = scinew GeomPoints();
	    sage_alg->search(isovals[iv], group, points);
	    geometry_handles.push_back( group );
	  }
	  sage_alg->release();
	}
      }
      break;
    default: // Error
      error("Unknown Algorithm requested.");
      return;
    }

    // Get the output field handle.
    if (gui_build_field_.get() && field_handles.size() && field_handles[0].get_rep()) {

      // Copy the name of field to the downstream field.
      string fldname;
      if (field_input_handle->get_property("name",fldname)) {
	for (unsigned int i=0; i < field_handles.size(); i++)
	  field_handles[i]->set_property("name",fldname, false);
      } else {
	for (unsigned int i=0; i < field_handles.size(); i++)
	  field_handles[i]->set_property("name", string("ExtractIsosurface"), false);
      }

      // Single field.
      if (field_handles.size() == 1)
	field_output_handle_ = field_handles[0];

      // Multiple field_handles.
      else {

	const TypeDescription *ftd = field_handles[0]->get_type_description();
	CompileInfoHandle ci = ExtractIsosurfaceAlgo::get_compile_info(ftd);
	
	Handle<ExtractIsosurfaceAlgo> algo;
	if (!module_dynamic_compile(ci, algo)) return;
	
	field_output_handle_ = algo->execute(field_handles);
      }

      // Get the output interpolant handle.
      if (build_interp) {
	if (interpolant_handles[0].get_rep()) {
	  if (interpolant_handles.size() == 1)
	    matrix_output_handle_ = interpolant_handles[0];
	  else
	    matrix_output_handle_ = append_sparse_matrices(interpolant_handles);
	}
	else
	  warning("Interpolant not computed for this input field type and data location.");
      } else
	matrix_output_handle_ = 0;

    } else {

      field_output_handle_ = 0;
      matrix_output_handle_ = 0;
    }

    // Merged the geometry results.
    if( gui_build_geom_.get() && isovals.size() ) {
      
      GeomGroup *geomGroup = scinew GeomGroup;

      for (unsigned int iv=0; iv<isovals.size(); iv++) {
	MaterialHandle material_handle;

	if (colormap_input_handle.get_rep())
	  material_handle = colormap_input_handle->lookup(isovals[iv]);
	else
	  material_handle = scinew Material(Color(gui_color_r_.get(),
					  gui_color_g_.get(),
					  gui_color_b_.get()));

	if (geometry_handles[iv].get_rep()) 
	  geomGroup->add(scinew GeomMaterial( geometry_handles[iv],
					      material_handle ));
      }

      if( geomGroup->size() ) {
	geometry_output_handle_ = GeomHandle( geomGroup );
      } else {
	delete geomGroup;
	geometry_output_handle_  = 0;
      }

    } else {
      geometry_output_handle_  = 0;
    }

    string fldname;
    if (!field_input_handle->get_property("name", fldname))
      fldname = string("ExtractIsosurface");

    send_output_handle( "Geometry", geometry_output_handle_, fldname );
  }
    
  // Send the isosurface field downstream
  send_output_handle( "Surface", field_output_handle_, true );
  send_output_handle( "Mapping", matrix_output_handle_, true );
}

} // End namespace SCIRun
