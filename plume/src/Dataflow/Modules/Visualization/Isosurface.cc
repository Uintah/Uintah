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


#include <map>
#include <iostream>
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

#include <Dataflow/Modules/Visualization/Isosurface.h>

//#include <typeinfo>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>
#include <Core/Containers/StringUtil.h>

#include <Dataflow/Network/Module.h>

namespace SCIRun {

DECLARE_MAKER(Isosurface)


Isosurface::Isosurface(GuiContext* ctx) : 
  Module("Isosurface", ctx, Filter, "Visualization", "SCIRun"), 
  gui_iso_value_min_(ctx->subVar("isoval-min")),
  gui_iso_value_max_(ctx->subVar("isoval-max")),
  gui_iso_value_(ctx->subVar("isoval")),
  gui_iso_value_typed_(ctx->subVar("isoval-typed")),
  gui_iso_value_quantity_(ctx->subVar("isoval-quantity")),
  gui_iso_quantity_range_(ctx->subVar("quantity-range")),
  gui_iso_quantity_clusive_(ctx->subVar("quantity-clusive")),
  gui_iso_quantity_min_(ctx->subVar("quantity-min")),
  gui_iso_quantity_max_(ctx->subVar("quantity-max")),
  gui_iso_quantity_list_(ctx->subVar("quantity-list")),
  gui_iso_value_list_(ctx->subVar("isoval-list")),
  gui_iso_matrix_list_(ctx->subVar("matrix-list")),
  gui_extract_from_new_field_(ctx->subVar("extract-from-new-field")),
  gui_use_algorithm_(ctx->subVar("algorithm")),
  gui_build_field_(ctx->subVar("build_trisurf")),
  gui_build_geom_(ctx->subVar("build_geom")),
  gui_np_(ctx->subVar("np")),
  gui_active_isoval_selection_tab_(ctx->subVar("active-isoval-selection-tab")),
  gui_active_tab_(ctx->subVar("active_tab")),
  gui_update_type_(ctx->subVar("update_type")),
  gui_color_r_(ctx->subVar("color-r")),
  gui_color_g_(ctx->subVar("color-g")),
  gui_color_b_(ctx->subVar("color-b")),
  use_algorithm_(-1),
  build_field_(-1),
  build_geom_(-1),
  np_(-1), 
  color_r_(-1),
  color_g_(-1),
  color_b_(-1),  
  fGeneration_(-1),
  cmGeneration_(-1),
  mGeneration_(-1),

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
  update_state(NeedData);
 
  bool update = false;

  FieldIPort *ifield_port = (FieldIPort *)get_iport("Field");
  FieldHandle fHandle;
  if (!(ifield_port->get(fHandle) && fHandle.get_rep())) {
    error( "No field handle or representation." );
    return;
  }

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {

    fGeneration_ = fHandle->generation;

    ScalarFieldInterfaceHandle sfi = fHandle->query_scalar_interface(this);
    if (!sfi.get_rep()) {
      error("Input field does not contain scalar data.");
      return;
    }

    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);
    if (minmax.first  != iso_value_min_ ||
	minmax.second != iso_value_max_) {

      iso_value_min_ = minmax.first;
      iso_value_max_ = minmax.second;

      ostringstream str;
      str << id << " set_min_max " << minmax.first << "  " << minmax.second;
      gui->execute(str.str().c_str());
    }

    if ( !gui_extract_from_new_field_.get() )
      return;

    update = true;
  }

  // Color the Geometry.
  ColorMapIPort *icmap_port = (ColorMapIPort *)get_iport("Optional Color Map");
  ColorMapHandle cmHandle;
  bool have_ColorMap = false;
  if (icmap_port->get(cmHandle)) {
    if(!cmHandle.get_rep()) {
      error( "No colormap representation." );
      return;
    }   
     
    have_ColorMap = true;
    if( cmGeneration_ != cmHandle->generation ) {
    
      cmGeneration_ = cmHandle->generation;
      update = true;
    }
  }
  
  vector<double> isovals(0);

  double qmax = iso_value_max_;
  double qmin = iso_value_min_;

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
      if (!have_ColorMap) {
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

    str << id << " set-isoquant-list \"";

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

    gui->execute(str.str().c_str());

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
	val = iso_value_min_ + (iso_value_max_ - iso_value_min_) * val / 100.0;
      }
      isovals.push_back(val);
    }
  } else if (gui_active_isoval_selection_tab_.get() == "3") { // matrix

    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Optional Isovalues");
    MatrixHandle mHandle;

    if (!imatrix_port->get(mHandle)) {
      gui->execute(id + " set-isomatrix-list \"No matrix present\"");
      error("Matrix selected - but no matrix is present.");
      return;
    } else if(!mHandle.get_rep()) {
      gui->execute(id + " set-isomatrix-list \"No matrix representation\"");
      error( "No matrix representation." );
      return;
    }

    if( mGeneration_ != mHandle->generation )
      mGeneration_ = mHandle->generation;
    
    ostringstream str;

    str << id << " set-isomatrix-list \"";

    for (int i=0; i < mHandle->nrows(); i++) {
      for (int j=0; j < mHandle->ncols(); j++) {
	isovals.push_back(mHandle->get(i, j));

	str << " " << isovals[i];
      }
    }

    str << "\"";

    gui->execute(str.str().c_str());

  } else {
    error("Bad active_isoval_selection_tab value");
    return;
  }

  // See if any of the isovalues have changed.
  if( isovals_.size() != isovals.size() ) {
    isovals_.resize( isovals.size() );
    update = true;
  }

  for( unsigned int i=0; i<isovals.size(); i++ ) {
    if( isovals_[i] != isovals[i] ) {
      isovals_[i] = isovals[i];
      update = true;
    }
  }

  int use_algorithm = gui_use_algorithm_.get();
  int build_field   = gui_build_field_.get();
  int build_geom    = gui_build_geom_.get();
  int np            = gui_np_.get();

  double color_r = gui_color_r_.get();
  double color_g = gui_color_g_.get();
  double color_b = gui_color_b_.get();
  
  if( use_algorithm_ != use_algorithm ||
      build_field_   != build_field ||
      build_geom_    != build_geom  ||
      np_            != np ||

      color_r_       != color_r  ||
      color_g_       != color_g  ||
      color_b_       != color_b ) {

    use_algorithm_ = use_algorithm;
    build_field_   = build_field;
    build_geom_    = build_geom;
    np_ = np;

    color_r_       = color_r;
    color_g_       = color_g;
    color_b_       = color_b;

    update = true;
  }

  // Decide if an interpolant will be computed for the output field.
  MatrixOPort *omatrix_port = (MatrixOPort *) get_oport("Mapping");

  const bool build_interp = build_field && omatrix_port->nconnections();

  if( (build_field  && !fHandle_.get_rep()) ||
      (build_interp && !mHandle_.get_rep()) ||
      (build_geom   && geomID_ == 0   ) ||
      update ||
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
	    mc_alg->search( isovals[iv], build_field, build_geom );
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
	    geometries.push_back(noise_alg->search(isovals[iv], build_field, build_geom));
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
    if (build_field && fields.size() && fields[0].get_rep()) {

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
	const TypeDescription *ftd = fields[0]->get_type_description(0);
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
      }
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

    if (build_geom) {
      // Merged send_results.
      GeomGroup *geom = scinew GeomGroup;;

      for (unsigned int iv=0; iv<isovals.size(); iv++) {
	MaterialHandle matl;

	if (have_ColorMap)
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

    update_state(Completed);
  }

  // Send the isosurface field downstream
  if( build_field && fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Surface");
    ofield_port->send( fHandle_ );
  }

  // Send the mapping matrix downstream
  if( build_interp && mHandle_.get_rep() )
  {
    MatrixOPort *omatrix_port = (MatrixOPort *) get_oport("Mapping");
    omatrix_port->send( mHandle_ );
  }
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
                       ftd->get_name() + "<double> ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}
} // End namespace SCIRun
