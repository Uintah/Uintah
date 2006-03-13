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
 *  FieldSubSample.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   January 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
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
  GuiInt power_app_;

  GuiInt Wrap_;
  GuiInt Dims_;

  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iStart_;
  GuiInt jStart_;
  GuiInt kStart_;

  GuiInt iStop_;
  GuiInt jStop_;
  GuiInt kStop_;

  GuiInt iStride_;
  GuiInt jStride_;
  GuiInt kStride_;

  GuiInt iWrap_;
  GuiInt jWrap_;
  GuiInt kWrap_;

  FieldHandle  fHandle_;
  MatrixHandle mHandle_;
};


DECLARE_MAKER(FieldSubSample)


FieldSubSample::FieldSubSample(GuiContext *context)
  : Module("FieldSubSample", context, Filter, "FieldsCreate", "SCIRun"),
    power_app_(context->subVar("power_app")),

    Wrap_(context->subVar("wrap"), 0 ),
    Dims_(context->subVar("dims"), 3 ),

    iDim_(context->subVar("i-dim"), 2),
    jDim_(context->subVar("j-dim"), 2),
    kDim_(context->subVar("k-dim"), 2),

    iStart_(context->subVar("i-start"), 0),
    jStart_(context->subVar("j-start"), 0),
    kStart_(context->subVar("k-start"), 0),

    iStop_(context->subVar("i-stop"), 1),
    jStop_(context->subVar("j-stop"), 1),
    kStop_(context->subVar("k-stop"), 1),

    iStride_(context->subVar("i-stride"), 1),
    jStride_(context->subVar("j-stride"), 1),
    kStride_(context->subVar("k-stride"), 1),
 
    iWrap_(context->subVar("i-wrap"), 0),
    jWrap_(context->subVar("j-wrap"), 0),
    kWrap_(context->subVar("k-wrap"), 0)
{
}


FieldSubSample::~FieldSubSample()
{
}


void
FieldSubSample::execute()
{
  update_state(NeedData);
  reset_vars();

  bool needToExecute = false;

  FieldHandle  fHandle;

  if( !getIHandle( "Input Field",  fHandle,  needToExecute, true  ) ) return;
  if( !getIHandle( "Input Matrix", mHandle_, needToExecute, false ) ) return;

  // The matrix is optional.
  if( mHandle_ != 0 &&
      (mHandle_->nrows() != 3 || mHandle_->ncols() != 5) ) {
    error( "Input matrix is not a 3x5 matrix" );
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

  if( fHandle->basis_order() != 0 &&
      fHandle->basis_order() != 1 ) {
    error( fHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Currently only availible for cell or node data." );
    return;
  }

  if( mesh_type.find("StructHexVolMesh"  ) != string::npos ||
      mesh_type.find("StructQuadSurfMesh") != string::npos ||
      mesh_type.find("StructCurveMesh"   ) != string::npos )
    Wrap_.set( 1, GuiVar::SET_GUI_ONLY );
  else
    Wrap_.set( 0, GuiVar::SET_GUI_ONLY );

  // Check to see if the gui dimensions are different than the field.
  if( Dims_.changed( true ) ||
      Wrap_.changed( true ) ||
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

    if( iDim_.get() != mHandle_->get(0, 4) ||
	jDim_.get() != mHandle_->get(1, 4) ||
	kDim_.get() != mHandle_->get(2, 4) ) {
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

    iStart_.set( (int) mHandle_->get(0, 0) );
    jStart_.set( (int) mHandle_->get(1, 0) );
    kStart_.set( (int) mHandle_->get(2, 0) );

    iStop_.set( (int) mHandle_->get(0, 1) );
    jStop_.set( (int) mHandle_->get(1, 1) );
    kStop_.set( (int) mHandle_->get(2, 1) );

    iStride_.set( (int) mHandle_->get(0, 2) );
    jStride_.set( (int) mHandle_->get(1, 2) );
    kStride_.set( (int) mHandle_->get(2, 2) );

    iWrap_.set( (int) mHandle_->get(0, 3) );
    jWrap_.set( (int) mHandle_->get(1, 3) );
    kWrap_.set( (int) mHandle_->get(2, 3) );
  }

  // Check to see if any values have changed via a matrix or user.
  if( iStart_.changed( true ) ||
      jStart_.changed( true ) ||
      kStart_.changed( true ) ||
      
      iStop_.changed( true ) ||
      jStop_.changed( true ) ||
      kStop_.changed( true ) ||
      
      iStride_.changed( true ) ||
      jStride_.changed( true ) ||
      kStride_.changed( true ) ||
      
      iWrap_.changed( true ) ||
      jWrap_.changed( true ) ||
      kWrap_.changed( true ) ) {

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
    CompileInfoHandle ci = FieldSubSampleAlgo::get_compile_info(ftd);
    Handle<FieldSubSampleAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fHandle_ = algo->execute(fHandle,
			     iStart_.get(),  jStart_.get(),  kStart_.get(),
			     iStop_.get(),   jStop_.get(),   kStop_.get(),
			     iStride_.get(), jStride_.get(), kStride_.get(),
			     iWrap_.get(),   jWrap_.get(),   kWrap_.get());
  }

  // Create the output matrix with the axis and index
  if( mHandle_ == 0 ) {
    DenseMatrix *selected = scinew DenseMatrix(3,5);

    selected->put(0, 0, iStart_.get() );
    selected->put(1, 0, jStart_.get() );
    selected->put(2, 0, kStart_.get() );

    selected->put(0, 1, iStop_.get() );
    selected->put(1, 1, jStop_.get() );
    selected->put(2, 1, kStop_.get() );

    selected->put(0, 2, iStride_.get() );
    selected->put(1, 2, jStride_.get() );
    selected->put(2, 2, kStride_.get() );

    selected->put(0, 3, iWrap_.get() );
    selected->put(1, 3, jWrap_.get() );
    selected->put(2, 3, kWrap_.get() );

    selected->put(0, 4, iDim_.get() );
    selected->put(1, 4, jDim_.get() );
    selected->put(2, 4, kDim_.get() );

    mHandle_ = MatrixHandle(selected);
  }

  // Send the data downstream
  setOHandle( "Output Field",  fHandle_, true );
  setOHandle( "Output Matrix", mHandle_, true );
}

CompileInfoHandle
FieldSubSampleAlgo::get_compile_info(const TypeDescription *ftd)
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
  rval->add_include(include_path);

  // Structured meshs have a set_point method which is
  // needed. However, it is not defined for gridded meshes. As such,
  // the include file defined below contains a compiler flag so that
  // when needed in FieldSlicer.h it is compiled.
  if( ftd->get_name().find("StructHexVolMesh"  ) != string::npos ||
      ftd->get_name().find("StructQuadSurfMesh") != string::npos ||
      ftd->get_name().find("StructCurveMesh"   ) != string::npos ) {

    string header_path(include_path);  // Get the right path 

    // Insert the Dynamic header file name.
    header_path.insert( header_path.find_last_of("."), "Dynamic" );

    rval->add_include(header_path);
  }

  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
