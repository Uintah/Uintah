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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Modules/Fields/FieldSlicer.h>
#include <Core/Basis/HexTrilinearLgn.h>


namespace SCIRun {

class FieldSlicer : public Module {
public:
  FieldSlicer(GuiContext *context);

  virtual ~FieldSlicer();

  virtual void execute();

private:
  GuiInt Axis_;
  GuiInt Dims_;

  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iIndex_;
  GuiInt jIndex_;
  GuiInt kIndex_;

  //updateType_ must be declared after all gui vars because some are
  //traced in the tcl code. If updateType_ is set to Auto having it
  //last will prevent the net from executing when it is instantiated.

  GuiString  updateType_;

  FieldHandle fHandle_;
  MatrixHandle mHandle_;

  int fGeneration_;
  int mGeneration_;
};


DECLARE_MAKER(FieldSlicer)


FieldSlicer::FieldSlicer(GuiContext *context)
  : Module("FieldSlicer", context, Filter, "FieldsCreate", "SCIRun"),
    
    Axis_(context->subVar("axis"), 2),
    Dims_(context->subVar("dims"), 3),

    iDim_(context->subVar("i-dim"), 1),
    jDim_(context->subVar("j-dim"), 1),
    kDim_(context->subVar("k-dim"), 1),

    iIndex_(context->subVar("i-index"), 1),
    jIndex_(context->subVar("j-index"), 1),
    kIndex_(context->subVar("k-index"), 1),

    updateType_(ctx->subVar("update_type"), "Manual"),

    fGeneration_(-1),
    mGeneration_(-1)
{
}


FieldSlicer::~FieldSlicer()
{
}


void
FieldSlicer::execute()
{
  update_state(NeedData);
  reset_vars();

  bool updateAll    = false;
  bool updateField  = false;
  bool updateMatrix = false;

  FieldHandle  fHandle;
  MatrixHandle mHandle;

  // Get a handle to the input field port.
  FieldIPort* ifield_port = (FieldIPort *) get_iport("Input Field");

  // The field input is required.
  if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
      !(fHandle->mesh().get_rep())) {
    error( "No field handle or representation" );
    return;
  }

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {
    fGeneration_ = fHandle->generation;
    updateField = true;
  }

  // Get a handle to the input matrix port.
  MatrixIPort* imatrix_port = (MatrixIPort *) get_iport("Input Matrix");

  // The matrix input is optional.
  if (imatrix_port->get(mHandle) && mHandle.get_rep()) {
    
    // Check to see if the input matrix has changed.
    if( mGeneration_ != mHandle->generation ) {
      mGeneration_ = mHandle->generation;
      updateMatrix = true;
    }

    if( mHandle->nrows() != 3 || mHandle->ncols() != 3 ) {
      error( "Input matrix is not a 3x3 matrix" );
      return;
    }

  } else {
    mGeneration_ = -1;
  }

  // Get the dimensions of the mesh.
  // this should be part of the dynamic compilation....
  string mesh_type =
    fHandle->get_type_description(Field::MESH_TD_E)->get_name();

  //FIX_ME MC how do i detect a "ITKLatVolField"
  if( mesh_type.find("LatVolMesh"      ) != string::npos ||
      mesh_type.find("StructHexVolMesh") != string::npos ) {
    typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
    LVMesh *lvmInput = (LVMesh*) fHandle->mesh().get_rep();

    iDim_.set_c( lvmInput->get_ni() );
    jDim_.set_c( lvmInput->get_nj() );
    kDim_.set_c( lvmInput->get_nk() );

    Dims_.set_c( 3 );

  } else if( mesh_type.find("ImageMesh"         ) != string::npos ||
	     mesh_type.find("StructQuadSurfMesh") != string::npos ) {
    typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
    IMesh *imInput = (IMesh*) fHandle->mesh().get_rep();
    iDim_.set_c( imInput->get_ni() );
    jDim_.set_c( imInput->get_nj() );
    kDim_.set_c( 1 );

    Dims_.set_c( 2 );

  } else if( mesh_type.find("ScanlineMesh"   ) != string::npos ||
	     mesh_type.find("StructCurveMesh") != string::npos ) {
    typedef ScanlineMesh<CrvLinearLgn<Point> > SLMesh;
    SLMesh *slmInput = (SLMesh*) fHandle->mesh().get_rep();
    
    iDim_.set_c( slmInput->get_ni() );
    jDim_.set_c( 1 );
    kDim_.set_c( 1 );

    Dims_.set_c( 1 );

  } else {
    error( fHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Only availible for regular topology e.g. uniformly gridded or structure gridded data." );
    return;
  }

  if( fHandle->basis_order() != 1 ) {
    error( fHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Currently only availible for node data." );
    return;
  }

  // Check to see if the dimensions have changed.
  if( Dims_.change() ||
      iDim_.change() ||
      jDim_.change() ||
      kDim_.change() )
  {
    // Dims has callback on it, so it must be set it after i, j, and k.
    ostringstream str;
    str << id
	<< " set_size "
	<< Dims_.get_c() << " "
	<< iDim_.get_c() << " "
	<< jDim_.get_c() << " "
	<< kDim_.get_c();
    gui->execute(str.str().c_str());

    updateAll = true;
  }

  if( mGeneration_ != -1 ) {

    if( iDim_.get_c() != mHandle->get(0, 2) ||
	jDim_.get_c() != mHandle->get(1, 2) ||
	kDim_.get_c() != mHandle->get(2, 2) ) {
      ostringstream str;
      str << "The dimensions of the matrix slicing do match the field. "
	  << " Expected "
	  << iDim_.get_c() << " "
	  << jDim_.get_c() << " "
	  << kDim_.get_c()
	  << " Got "
	  << mHandle->get(0, 2) << " "
	  << mHandle->get(1, 2) << " "
	  << mHandle->get(2, 2);
      
      error( str.str() );
      return;
    }

    int axis = -1;

    for (int i=0; i < mHandle->nrows(); i++)
      if( mHandle->get(i, 0) == 1 )
	axis = i;

    if( axis == -1 ) {
      ostringstream str;
      str << "The input slicing matrix has no axis selected. ";      
      error( str.str() );
      return;
    }

    ostringstream str;
    str << id << " set_index "
	<< axis << " "
	<< (int) mHandle->get(0, 1) << " "
	<< (int) mHandle->get(1, 1) << " "
	<< (int) mHandle->get(2, 1);

    gui->execute(str.str().c_str());

    reset_vars();
  }

  // Check to see if the gui values have changed.
  if( iIndex_.change( true ) ||
      jIndex_.change( true ) ||
      kIndex_.change( true ) ||
      Axis_.change( true ))
    updateAll = true;

  if( mGeneration_ == -1 ) {
    DenseMatrix *selected = scinew DenseMatrix(3,3);

    for (int i=0; i < 3; i++)
      selected->put(i, 0, (double) (Axis_.get_c() == i) );

    selected->put(0, 1, iIndex_.get_c() );
    selected->put(1, 1, jIndex_.get_c() );
    selected->put(2, 1, kIndex_.get_c() );

    selected->put(0, 2, iDim_.get_c() );
    selected->put(1, 2, jDim_.get_c() );
    selected->put(2, 2, kDim_.get_c() );

    mHandle_ = MatrixHandle(selected);
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || updateAll || updateField || updateMatrix )
  {
    const TypeDescription *ftd = fHandle->get_type_description();
    const TypeDescription *ttd = fHandle->get_type_description(Field::FDATA_TD_E);

    CompileInfoHandle ci = FieldSlicerAlgo::get_compile_info(ftd,ttd);
    Handle<FieldSlicerAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    unsigned int index;
    if (Axis_.get_c() == 0) {
      index = iIndex_.get_c();
    }
    else if (Axis_.get_c() == 1) {
      index = jIndex_.get_c();
    }
    else {
      index = kIndex_.get_c();
    }

    fHandle_ = algo->execute(fHandle, Axis_.get_c());

    // Now the new field is defined so do the work on it.
    const TypeDescription *iftd = fHandle->get_type_description();
    const TypeDescription *oftd = fHandle_->get_type_description();

    ci = FieldSlicerWorkAlgo::get_compile_info(iftd,oftd);
    Handle<FieldSlicerWorkAlgo> workalgo;

    if (!module_dynamic_compile(ci, workalgo)) return;
  
    workalgo->execute(fHandle, fHandle_, index, Axis_.get_c());
  }

  // Send the data downstream
  if( fHandle_.get_rep() ) {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output Field");

    if (!ofield_port) {
      error("Unable to initialize oport 'Output Field'.");
      return;
    }

    ofield_port->send_and_dereference( fHandle_, true );
  }

  // Get a handle to the output double port.
  if( mHandle_.get_rep() ) {
    MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");

    if (!omatrix_port) {
      error("Unable to initialize oport 'Output Matrix'.");
      return;
    }

    omatrix_port->send_and_dereference(mHandle_, true);
  }
}

CompileInfoHandle
FieldSlicerAlgo::get_compile_info(const TypeDescription *ftd,
				  const TypeDescription *ttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
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

  // Add in the include path to compile this obj
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
				      const TypeDescription *oftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
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

  // Add in the include path to compile this obj
  rval->add_include(include_path);  

  // Structured meshs have a set_point method which is
  // needed. However, it is not defined for gridded meshes. As such,
  // the include file defined below contains a compiler flag so that
  // when needed in FieldSlicer.h it is compiled.
  if( iftd->get_name().find("StructHexVolMesh"  ) != string::npos ||
      iftd->get_name().find("StructQuadSurfMesh") != string::npos ||
      iftd->get_name().find("StructCurveMesh"   ) != string::npos ||
      iftd->get_name().find("PointCloudMesh"    ) != string::npos )
  {
    string header_path(include_path);  // Get the right path 

    // Insert the Dynamic header file name.
    header_path.insert( header_path.find_last_of("."), "Dynamic" );

    rval->add_include(header_path);
  }

  iftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
