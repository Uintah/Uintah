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

#include <Packages/Fusion/Dataflow/Modules/Fields/IsoValueController.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Containers/StringUtil.h>


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

  // Get the current original field.
  if( !getIHandle( "Input Original Field", fHandle_orig_, true ) ) return;

  // Get the current transformed field.
  if( !getIHandle( "Input Transformed Field", fHandle_tran_, true ) ) return;

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
    inputs_changed_ = true;

    isovalues_.resize(isovalues.size());

    for( unsigned int i=0; i<isovalues.size(); i++ )
      isovalues_[i] == isovalues[i];

  } else {
    for( unsigned int i=0; i<isovalues.size(); i++ )
      if( fabs( isovalues_[i] - isovalues[i] ) > 1.0e-4 ) {
	inputs_changed_ = true;
	break;
      }
  }

  vector< FieldHandle > fHandles_N_1D;
  vector< FieldHandle > fHandles_ND;
  vector< GeomHandle  > gHandles;

  if( inputs_changed_ || error_ ) {
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

      bool intermediate = ( i<isovalues.size()-1 );

      sendOHandle( "Output Original Field",    fHandle_orig_,    true, intermediate );
      sendOHandle( "Output Transformed Field", fHandle_tran_,    true, intermediate );
      sendOHandle( "Isovalue",                 mHandleIsoValue_, true, intermediate );
      sendOHandle( "Index",                    mHandleIndex_,    true, intermediate );

      str << "  Done";

      remark( str.str());

      // Get the original isosurfaces.
      FieldHandle fHandleND;

      if( !getIHandle( "(N)D Field", fHandleND, true ) ) {
	error_ = true;
	return;
      } else
	fHandles_ND.push_back( fHandleND );

      // Get the transformed isosurfaces.
      FieldHandle fHandleN_1D;
           
      if( !getIHandle( "(N-1)D Field", fHandleN_1D, true ) ) {
	error_ = true;
	return;
      } else
	fHandles_N_1D.push_back( fHandleN_1D );


      // Get the transformed geometry.
//    GeomHandle geometryin;

//    if( !getIHandle( "Axis Geometry", geometryin, true ) ) {
// 	error_ = true;
//      return;
//    } else
//  	gHandles.push_back( geometryin );
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

  // Send the data downstream
  sendOHandle( "(N)D Fields",   fHandle_ND_,   true );
  sendOHandle( "(N-1)D Fields", fHandle_N_1D_, true );

  sendOHandle( "Axis Geometry", gHandles, fldname );
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
