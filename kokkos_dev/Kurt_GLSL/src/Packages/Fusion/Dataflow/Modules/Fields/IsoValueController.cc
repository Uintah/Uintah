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
 *  IsoValueController.cc: Send a serries of Isovalues to the IsoSurface module
 *                         and collect the resulting surfaces.
 *
 *  Written by:
 *   Allen R. Sanderson
 *   University of Utah
 *   August 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Datatypes/FieldInterface.h>

#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/CurveMesh.h>

#include <Core/Containers/StringUtil.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/IsoValueController.h>

namespace Fusion {

using namespace SCIRun;

DECLARE_MAKER(IsoValueController)
IsoValueController::IsoValueController(GuiContext* context)
  : Module("IsoValueController", context, Source, "Fields", "Fusion"),
    IsoValueStr_(context->subVar("isovalues")),
    prev_min_(0),
    prev_max_(0),
    last_orig_generation_(-1),
    last_tran_generation_(-1),
    error_(false)
{
  ColumnMatrix *isovalueMtx = scinew ColumnMatrix(1);
  mHandleIsoValue_ = MatrixHandle(isovalueMtx);
  
  ColumnMatrix *indexMtx = scinew ColumnMatrix(2);
  mHandleIndex_ = MatrixHandle(indexMtx);
}

IsoValueController::~IsoValueController(){
}

void IsoValueController::execute() {

  // Get a handle to the input original field port.
  FieldIPort *ifpo = (FieldIPort *)get_iport("Input Original Field");
  if (!ifpo) {
    error("Unable to initialize iport 'Input Original Field'.");
    return;
  }

  // Get a handle to the input transformed field port.
  FieldIPort *ifpt = (FieldIPort *)get_iport("Input Transformed Field");
  if (!ifpt) {
    error("Unable to initialize iport 'Input Transformed Field'.");
    return;
  }

  // Get a handle to the input (N-1)D field port.
  FieldIPort *ifieldN_1D_port = (FieldIPort *)get_iport("(N-1)D Field");
  if (!ifieldN_1D_port) {
    error("Unable to initialize oport '(N-1)D Field'.");
    return;
  }

  // Get a handle to the input (N)D field port.
  FieldIPort *ifieldND_port = (FieldIPort *)get_iport("(N)D Field");
  if (!ifieldND_port) {
    error("Unable to initialize oport '(N)D Field'.");
    return;
  }

  // Get a handle to the input geometery port.
  GeometryIPort *igeometry_port = (GeometryIPort *)get_iport("Axis Geometry");
  if (!igeometry_port) {
    error("Unable to initialize oport 'Axis Geometry'.");
    return;
  }


  // Get a handle to the output original field port.
  FieldOPort *ofpo = (FieldOPort *)get_oport("Output Original Field");
  if (!ofpo) {
    error("Unable to initialize iport 'Output Original Field'.");
    return;
  }

  // Get a handle to the output transformed field port.
  FieldOPort *ofpt = (FieldOPort *)get_oport("Output Transformed Field");
  if (!ofpt) {
    error("Unable to initialize iport 'Output Transformed Field'.");
    return;
  }

  // Get a handle to the output isovalue port.
  MatrixOPort *omatrixIsovalue_port = (MatrixOPort *)get_oport("Isovalue");
  if (!omatrixIsovalue_port) {
    error("Unable to initialize oport 'Isovalue'.");
    return;
  }

  // Get a handle to the output index port.
  MatrixOPort *omatrixIndex_port = (MatrixOPort *)get_oport("Index");
  if (!omatrixIndex_port) {
    error("Unable to initialize oport 'Index'.");
    return;
  }


  bool update = false;

  // Get the current original field.
  if (!(ifpo->get(fHandle_orig_) && fHandle_orig_.get_rep())) {
    error( "No field handle or representation." );
    return;
  }

  if ( fHandle_orig_->generation != last_orig_generation_ ) {
    last_orig_generation_ = fHandle_orig_->generation;
    update = true;
  }

  // Get the current transformed field.
  if (!(ifpt->get(fHandle_tran_) && fHandle_tran_.get_rep())) {
    error( "No field handle or representation." );
    return;
  }

  if ( fHandle_tran_->generation != last_tran_generation_ ) {
    // new field
    ScalarFieldInterfaceHandle sfi =
      fHandle_tran_->query_scalar_interface(this);
    if (!sfi.get_rep()) {
      error("Input field does not contain scalar data.");
      return;
    }
    
    // Set min/max
    pair<double, double> minmax;
    sfi->compute_min_max(minmax.first, minmax.second);
    if (minmax.first != prev_min_ || minmax.second != prev_max_) {
      prev_min_ = minmax.first;
      prev_max_ = minmax.second;
    }

    last_tran_generation_ = fHandle_tran_->generation;
    update = true;
  }

  vector< double > isovalues(0);

  istringstream vlist(IsoValueStr_.get());
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
      val = prev_min_ + (prev_max_ - prev_min_) * val / 100.0;
    }

    if( prev_min_ <= val && val <= prev_max_ )
      isovalues.push_back(val);
    else {
      error("Isovalue is less than the minimum or greater than the maximum.");
      ostringstream str;
      str << "Minimum: " << prev_min_ << " Maximum: " << prev_max_;
    
      error( str.str());
      return;
    }
  }

  if( isovalues_.size() != isovalues.size() ){
    update = true;

    isovalues_.resize(isovalues.size());

    for( unsigned int i=0; i<isovalues.size(); i++ )
      isovalues_[i] == isovalues[i];

  } else {
    for( unsigned int i=0; i<isovalues.size(); i++ )
      if( fabs( isovalues_[i] - isovalues[i] ) > 1.0e-4 ) {
	update = true;
	break;
      }
  }

  vector< FieldHandle > fHandles_N_1D;
  vector< FieldHandle > fHandles_ND;
  vector< GeomHandle  > gHandles;

  if( update || error_ ) {
    error_ = false;

    unsigned int ncols = (int) sqrt( (double) isovalues.size() );

    if( ncols * ncols < isovalues.size() )
      ncols++;

    unsigned int nrows = isovalues.size() / ncols;

    if( nrows * ncols < isovalues.size() )
      nrows++;

    int row = 0;
    unsigned int col = 0;

    for( unsigned int i=0; i<isovalues.size(); i++ ) {

      mHandleIsoValue_.get_rep()->put(0, 0, isovalues[i]);

      mHandleIndex_.get_rep()->put(0, 0, (double) col);
      mHandleIndex_.get_rep()->put(1, 0, (double) row);

      ostringstream str;
      str << "Using Isovalue " << isovalues[i];
      str << "  Row " << row << " Col " << col;
    
      if( ++col == ncols ) {
	col = 0;
	--row;
      }

      if( i<isovalues.size()-1 ) {
	ofpo->send_intermediate(fHandle_orig_);
	ofpt->send_intermediate(fHandle_tran_);
	omatrixIsovalue_port->send_intermediate(mHandleIsoValue_);
	omatrixIndex_port->send_intermediate(mHandleIndex_);

      } else {
	ofpo->send(fHandle_orig_);
	ofpt->send(fHandle_tran_);
	omatrixIsovalue_port->send(mHandleIsoValue_);
	omatrixIndex_port->send(mHandleIndex_);
	str << "  Done";
      }

      remark( str.str());

      // Get the original isosurfaces.
      FieldHandle fHandleND;

      if (!(ifieldND_port->get(fHandleND) && fHandleND.get_rep())) {
	error( "No (N)D field handle or representation." );
	error_ = true;
	return;
      } else
	fHandles_ND.push_back( fHandleND );

      // Get the transformed isosurfaces.
      FieldHandle fHandleN_1D;
           
      if (!(ifieldN_1D_port->get(fHandleN_1D) && fHandleN_1D.get_rep())) {
	error( "No (N-1)D field handle or representation." );
	error_ = true;
	return;
      } else
	fHandles_N_1D.push_back( fHandleN_1D );

//    GeomHandle geometryin;
                
//    if (!(igeometry_port->getObj(geometryin) && geometryin.get_rep())) {
// 	error( "No geometry handle or representation." );
// 	return;
//    } else
// 	gHandles.push_back( geometryin );
    }
  }

  // Copy the name of field to the downstream field.
  string fldname;
  if (!fHandle_orig_->get_property("name",fldname))
    fldname = string("Isosurface");


  // Output field.
  if (fHandles_ND.size() && fHandles_ND[0].get_rep()) {

    for (unsigned int i=0; i<fHandles_ND.size(); i++) {
      fHandles_ND[i]->set_property("name",fldname, false);
      fHandles_N_1D[i]->set_property("name",fldname, false);
    }

    // Single field.
    if (fHandles_N_1D.size() == 1) {

      fHandle_ND_   = fHandles_ND[0];
      fHandle_N_1D_ = fHandles_N_1D[0];

    // Multiple fields.
    } else {
      const TypeDescription *ftd = fHandles_N_1D[0]->get_type_description();
      CompileInfoHandle ci = IsoValueControllerAlgo::get_compile_info(ftd);
	
      Handle<IsoValueControllerAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;
	
      algo->execute(fHandles_ND, fHandles_N_1D,
		    fHandle_ND_, fHandle_N_1D_ );
    }
  }

  if (fHandles_ND.size() && fHandles_ND[0].get_rep()) {

    FieldOPort *ofieldND_port = (FieldOPort *)get_oport("(N)D Fields");
    if (!ofieldND_port) {
      error("Unable to initialize oport '(N)D Fields'.");
      return;
    }

    FieldOPort *ofieldN_1D_port = (FieldOPort *)get_oport("(N-1)D Fields");
    if (!ofieldN_1D_port) {
      error("Unable to initialize oport '(N-1)D Fields'.");
      return;
    }
  
    ofieldND_port->send(fHandle_ND_);
    ofieldN_1D_port->send(fHandle_N_1D_);
  }

  if( gHandles.size() ) {
    GeometryOPort *ogeometry_port =
      (GeometryOPort *)get_oport("Axis Geometry");

    if (!ogeometry_port) {
      error("Unable to initialize oport 'Axis Geometry'.");
      return;
    }    
    
    for (unsigned int i=0; i<gHandles.size(); i++)
      ogeometry_port->addObj(gHandles[i],fldname);
    
    if (gHandles.size())
      ogeometry_port->flushViews();
  }
}

void
IsoValueController::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

CompileInfoHandle
IsoValueControllerAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("IsoValueControllerAlgoT");
  static const string base_class_name("IsoValueControllerAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  rval->add_namespace("Fusion");
  return rval;
}
} // End namespace Fusion
