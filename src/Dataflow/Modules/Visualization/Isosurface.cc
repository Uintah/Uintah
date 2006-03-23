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

//    File   : Isosurface.cc
//    Author : Yarden Livnat
//    Date   : Fri Jun 15 16:38:02 2001


#include <Dataflow/Modules/Visualization/Isosurface.h>

#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>

#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>
#include <Core/Containers/StringUtil.h>

#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>


namespace SCIRun {

DECLARE_MAKER(Isosurface)


Isosurface::Isosurface(GuiContext* ctx) : 
  Module("Isosurface", ctx, Filter, "Visualization", "SCIRun"), 
  gui_iso_value_min_(get_ctx()->subVar("isoval-min"),  0.0),
  gui_iso_value_max_(get_ctx()->subVar("isoval-max"), 99.0),
  gui_iso_value_(get_ctx()->subVar("isoval"), 0.0),
  gui_iso_value_typed_(get_ctx()->subVar("isoval-typed"), 0.0),
  gui_iso_value_quantity_(get_ctx()->subVar("isoval-quantity"), 1),
  gui_iso_quantity_range_(get_ctx()->subVar("quantity-range"), "field"),
  gui_iso_quantity_clusive_(get_ctx()->subVar("quantity-clusive"), "exclusive"),
  gui_iso_quantity_min_(get_ctx()->subVar("quantity-min"),   0),
  gui_iso_quantity_max_(get_ctx()->subVar("quantity-max"), 100),
  gui_iso_quantity_list_(get_ctx()->subVar("quantity-list"), ""),
  gui_iso_value_list_(get_ctx()->subVar("isoval-list"), "No values present."),
  gui_iso_matrix_list_(get_ctx()->subVar("matrix-list"), "No matrix present - execution needed."),
  gui_extract_from_new_field_(get_ctx()->subVar("extract-from-new-field"), 1),
  gui_use_algorithm_(get_ctx()->subVar("algorithm"), 0),
  gui_build_field_(get_ctx()->subVar("build_trisurf"), 1),
  gui_build_geom_(get_ctx()->subVar("build_geom"), 1),
  gui_np_(get_ctx()->subVar("np"), 1),
  gui_active_isoval_selection_tab_(get_ctx()->subVar("active-isoval-selection-tab"), "0"),
  gui_active_tab_(get_ctx()->subVar("active_tab"), "0"),
  gui_update_type_(get_ctx()->subVar("update_type"), "On Release"),
  gui_color_r_(get_ctx()->subVar("color-r"), 0.4),
  gui_color_g_(get_ctx()->subVar("color-g"), 0.2),
  gui_color_b_(get_ctx()->subVar("color-b"), 0.9),
  fHandle_(0),
  mHandle_(0),
  geomID_(0),

  error_(0)  
{
}


Isosurface::~Isosurface()
{
}


MatrixHandle
append_sparse(vector<MatrixHandle> &matrices)
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
Isosurface::execute()
{
  FieldHandle fHandle;
  if( !get_input_handle( "Field", fHandle, true ) ) return;

  // Check to see if the input field has changed.
  if(inputs_changed_ ) {

    ScalarFieldInterfaceHandle sfi = fHandle->query_scalar_interface(this);
    if (!sfi.get_rep()) {
      error("Input field does not contain scalar data.");
      return;
    }

    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);

    gui_iso_value_min_.set( minmax.first,  GuiVar::SET_GUI_ONLY );
    gui_iso_value_max_.set( minmax.second, GuiVar::SET_GUI_ONLY );

    // Check to see if the gui min max are different than the field.
    if (gui_iso_value_min_.changed( true ) ||
	gui_iso_value_min_.changed( true ) ) {

      ostringstream str;
      str << get_id() << " set_min_max ";
      get_gui()->execute(str.str().c_str());
    }

    if ( !gui_extract_from_new_field_.get() )
      return;
  }

  // Get the optional colormap for the geometry.
  ColorMapHandle cmHandle = 0;
  if( !get_input_handle( "Optional Color Map", cmHandle, false ) ) return;
  
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
      error("Isosurface quantity must be at least one -- skipping isosurfacing.");
      return;
    }

    string range = gui_iso_quantity_range_.get();

    if (range == "colormap") {
      if (cmHandle != 0 ) {
	error("No color colormap for isovalue quantity");
	return;
      }
      qmin = cmHandle->getMin();
      qmax = cmHandle->getMax();
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

    MatrixHandle mHandle;
    if( !get_input_handle( "Optional Isovalues", mHandle, true ) ) return;

    ostringstream str;
    
    str << get_id() << " set-isomatrix-list \"";

    for (int i=0; i < mHandle->nrows(); i++) {
      for (int j=0; j < mHandle->ncols(); j++) {
	isovals.push_back(mHandle->get(i, j));

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

  if( (gui_build_field_.get() && !fHandle_.get_rep()) ||
      (gui_build_geom_.get()  && geomID_ == 0   ) ||
      (build_interp           && !mHandle_.get_rep()) ||
      inputs_changed_ ||
      error_ ) {

    update_state(JustStarted);

    error_ = false;

    vector<GeomHandle > geometries;
    vector<FieldHandle> fields;
    vector<MatrixHandle> interpolants;

    const TypeDescription *td = fHandle->get_type_description();

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
	  if (np < 1 ) { np = 1; gui_np_.set(np); }
	  if (np > 32 ) { np = 32; gui_np_.set(np); }
	  mc_alg->set_np(np);
	  mc_alg->set_field( fHandle.get_rep() );

	  for (unsigned int iv=0; iv<isovals.size(); iv++)  {
	    mc_alg->search( isovals[iv],
			    gui_build_field_.get(), gui_build_geom_.get() );
	    geometries.push_back( mc_alg->get_geom() );
	    for (int i = 0 ; i < np; i++) {
	      fields.push_back( mc_alg->get_field(i) );
	      if (build_interp)
		interpolants.push_back( mc_alg->get_interpolant(i) );
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
				       fHandle->basis_order() == 0,
				       false);
	  if (! module_dynamic_compile(ci, noise_alg)) {
	    error( "NOISE can not work with this field.");
	    return;
	  }
	  noise_alg->set_field(fHandle.get_rep());

	  for (unsigned int iv=0; iv<isovals.size(); iv++) {
	    geometries.push_back(noise_alg->search(isovals[iv],
			   gui_build_field_.get(), gui_build_geom_.get() ) );
	    fields.push_back(noise_alg->get_field());
	    if (build_interp)
	      interpolants.push_back(noise_alg->get_interpolant());
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
	  sage_alg->set_field(fHandle.get_rep());

	  for (unsigned int iv=0; iv<isovals.size(); iv++) {
	    GeomGroup *group = scinew GeomGroup;
	    GeomPoints *points = scinew GeomPoints();
	    sage_alg->search(isovals[0], group, points);
	    geometries.push_back( group );
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
    if (gui_build_field_.get() && fields.size() && fields[0].get_rep()) {

      // Copy the name of field to the downstream field.
      string fldname;
      if (fHandle->get_property("name",fldname)) {
	for (unsigned int i=0; i < fields.size(); i++)
	  fields[i]->set_property("name",fldname, false);
      } else {
	for (unsigned int i=0; i < fields.size(); i++)
	  fields[i]->set_property("name", string("Isosurface"), false);
      }

      // Single field.
      if (fields.size() == 1)
	fHandle_ = fields[0];

      // Multiple fields.
      else {

	const TypeDescription *ftd = fields[0]->get_type_description();
	CompileInfoHandle ci = IsosurfaceAlgo::get_compile_info(ftd);
	
	Handle<IsosurfaceAlgo> algo;
	if (!module_dynamic_compile(ci, algo)) return;
	
	fHandle_ = algo->execute(fields);
      }

      // Get the output interpolant handle.
      if (build_interp) {
	if (interpolants[0].get_rep()) {
	  if (interpolants.size() == 1)
	    mHandle_ = interpolants[0];
	  else
	    mHandle_ = append_sparse(interpolants);
	}
	else
	  warning("Interpolant not computed for this input field type and data location.");
      } else
	mHandle_ = 0;

    } else {

      fHandle_ = 0;
      mHandle_ = 0;
    }
  
    // Output geometry.
    GeometryOPort *ogeom_port = (GeometryOPort *)get_oport("Geometry");

    // Stop showing the previous geometry.
    bool geomflush = false;

    if ( geomID_ ) {
      ogeom_port->delObj( geomID_ );
      geomID_ = 0;
      geomflush = true;
    }

    if ( gui_build_geom_.get() ) {
      // Merged send_results.
      GeomGroup *geom = scinew GeomGroup;;

      for (unsigned int iv=0; iv<isovals.size(); iv++) {
	MaterialHandle matl;

	if (cmHandle != 0)
	  matl= cmHandle->lookup(isovals[iv]);
	else
	  matl = scinew Material(Color(gui_color_r_.get(),
				       gui_color_g_.get(),
				       gui_color_b_.get()));

	if (geometries[iv].get_rep()) 
	  geom->add(scinew GeomMaterial( geometries[iv] , matl ));
      }

      if (!geom->size())
	delete geom;
      else {
	string fldname;
	if (fHandle->get_property("name",fldname))
	  geomID_ = ogeom_port->addObj( geom, fldname );
	else 
	  geomID_ = ogeom_port->addObj( geom, string("Isosurface") );
	
	geomflush = true;
      }
    }

    if (geomflush)
      ogeom_port->flushViews();
  }
    
  // Send the isosurface field downstream
  send_output_handle( "Surface", fHandle_, true );
  send_output_handle( "Mapping", mHandle_, true );
}

CompileInfoHandle
IsosurfaceAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("IsosurfaceAlgoT");
  static const string base_class_name("IsosurfaceAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}
} // End namespace SCIRun
