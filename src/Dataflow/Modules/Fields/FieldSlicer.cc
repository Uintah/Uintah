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

    updateType_(ctx->subVar("update_type"), "Manual")
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

  bool needToExecute = false;

  FieldHandle  fHandle;

  if( !getIHandle( "Input Field",  fHandle,  needToExecute, true  ) ) return;
  if( !getIHandle( "Input Matrix", mHandle_, needToExecute, false ) ) return;

  // The matrix is optional.
  if( mHandle_ != 0 &&
      (mHandle_->nrows() != 3 || mHandle_->ncols() != 3) ) {
    error( "Input matrix is not a 3x3 matrix" );
    return;
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

    iDim_.set( lvmInput->get_ni(), GuiVar::SET_GUI_ONLY );
    jDim_.set( lvmInput->get_nj(), GuiVar::SET_GUI_ONLY );
    kDim_.set( lvmInput->get_nk(), GuiVar::SET_GUI_ONLY );

    Dims_.set( 3, GuiVar::SET_GUI_ONLY );

  } else if( mesh_type.find("ImageMesh"         ) != string::npos ||
	     mesh_type.find("StructQuadSurfMesh") != string::npos ) {
    typedef ImageMesh<QuadBilinearLgn<Point> > IMesh;
    IMesh *imInput = (IMesh*) fHandle->mesh().get_rep();
    iDim_.set( imInput->get_ni(), GuiVar::SET_GUI_ONLY );
    jDim_.set( imInput->get_nj(), GuiVar::SET_GUI_ONLY );
    kDim_.set( 1, GuiVar::SET_GUI_ONLY );

    Dims_.set( 2, GuiVar::SET_GUI_ONLY );

  } else if( mesh_type.find("ScanlineMesh"   ) != string::npos ||
	     mesh_type.find("StructCurveMesh") != string::npos ) {
    typedef ScanlineMesh<CrvLinearLgn<Point> > SLMesh;
    SLMesh *slmInput = (SLMesh*) fHandle->mesh().get_rep();
    
    iDim_.set( slmInput->get_ni(), GuiVar::SET_GUI_ONLY );
    jDim_.set( 1, GuiVar::SET_GUI_ONLY );
    kDim_.set( 1, GuiVar::SET_GUI_ONLY );

    Dims_.set( 1, GuiVar::SET_GUI_ONLY );

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

  // Check to see if the gui dimensions are different than the field.
  if( Dims_.changed( true ) ||
      iDim_.changed( true ) ||
      jDim_.changed( true ) ||
      kDim_.changed( true ) )
  {
    // Dims has callback on it, so it must be set it after i, j, and k.
    ostringstream str;
    str << id << " set_size ";
    gui->execute(str.str().c_str());

    reset_vars();
  }

  // An input matrix is present so use the values in it to override
  // the variables set in the gui.
  if( mHandle_ != 0 ) {

    if( iDim_.get() != mHandle_->get(0, 2) ||
	jDim_.get() != mHandle_->get(1, 2) ||
	kDim_.get() != mHandle_->get(2, 2) ) {
      ostringstream str;
      str << "The dimensions of the matrix slicing do match the field. "
	  << " Expected "
	  << iDim_.get() << " "
	  << jDim_.get() << " "
	  << kDim_.get()
	  << " Got "
	  << mHandle_->get(0, 2) << " "
	  << mHandle_->get(1, 2) << " "
	  << mHandle_->get(2, 2);
      
      error( str.str() );
      return;
    }

    // Check to see what axis has been selected.
    Axis_.set( -1 );

    for (int i=0; i < mHandle_->nrows(); i++)
      if( mHandle_->get(i, 0) == 1 )
	Axis_.set( i );

    if( Axis_.get() == -1 ) {
      ostringstream str;
      str << "The input slicing matrix has no axis selected. ";      
      error( str.str() );
      return;
    }

    iIndex_.set( (int) mHandle_->get(0, GuiVar::SET_GUI_ONLY) );
    jIndex_.set( (int) mHandle_->get(1, GuiVar::SET_GUI_ONLY) );
    kIndex_.set( (int) mHandle_->get(2, GuiVar::SET_GUI_ONLY) );
  }

  // Check to see if any values have changed via a matrix or user.
  if( Axis_.changed( true ) ||
      (Axis_.get() == 0 && iIndex_.changed( true )) ||
      (Axis_.get() == 1 && jIndex_.changed( true )) ||
      (Axis_.get() == 2 && kIndex_.changed( true )) ) {

    needToExecute = true;

    if( mHandle_ != 0 ) {
      ostringstream str;
      str << id << " update_index ";
      
      gui->execute(str.str().c_str());
      
      reset_vars();
    }
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || needToExecute ) {
    const TypeDescription *ftd = fHandle->get_type_description();
    const TypeDescription *ttd = fHandle->get_type_description(Field::FDATA_TD_E);

    CompileInfoHandle ci = FieldSlicerAlgo::get_compile_info(ftd,ttd);
    Handle<FieldSlicerAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    unsigned int index;
    if (Axis_.get() == 0) {
      index = iIndex_.get();
    }
    else if (Axis_.get() == 1) {
      index = jIndex_.get();
    }
    else {
      index = kIndex_.get();
    }

    fHandle_ = algo->execute(fHandle, Axis_.get());

    // Now the new field is defined so do the work on it.
    const TypeDescription *iftd = fHandle->get_type_description();
    const TypeDescription *oftd = fHandle_->get_type_description();

    ci = FieldSlicerWorkAlgo::get_compile_info(iftd,oftd);
    Handle<FieldSlicerWorkAlgo> workalgo;

    if (!module_dynamic_compile(ci, workalgo)) return;
  
    workalgo->execute(fHandle, fHandle_, index, Axis_.get());
  }

  // Create the output matrix with the axis and index
  if( mHandle_ == 0 ) {
    DenseMatrix *selected = scinew DenseMatrix(3,3);

    for (int i=0; i < 3; i++)
      selected->put(i, 0, (double) (Axis_.get() == i) );

    selected->put(0, 1, iIndex_.get() );
    selected->put(1, 1, jIndex_.get() );
    selected->put(2, 1, kIndex_.get() );

    selected->put(0, 2, iDim_.get() );
    selected->put(1, 2, jDim_.get() );
    selected->put(2, 2, kDim_.get() );

    mHandle_ = MatrixHandle(selected);
  }

  // Send the data downstream
  setOHandle( "Output Field",  fHandle_, true );
  setOHandle( "Output Matrix", mHandle_, true );
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
