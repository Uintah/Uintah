/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Modules/Fields/FieldSubSample.h>

namespace SCIRun {

class FieldSubSample : public Module {
public:
  FieldSubSample(GuiContext *context);

  virtual ~FieldSubSample();

  virtual void execute();

private:
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

  int idim_;
  int jdim_;
  int kdim_;

  int istart_;
  int jstart_;
  int kstart_;

  int istop_;
  int jstop_;
  int kstop_;

  int istride_;
  int jstride_;
  int kstride_;

  int iwrap_;
  int jwrap_;
  int kwrap_;

  FieldHandle  fHandle_;

  int fGeneration_;
  int mGeneration_;
};


DECLARE_MAKER(FieldSubSample)


FieldSubSample::FieldSubSample(GuiContext *context)
  : Module("FieldSubSample", context, Filter, "FieldsCreate", "SCIRun"),

    Wrap_(context->subVar("wrap")),
    Dims_(context->subVar("dims")),

    iDim_(context->subVar("i-dim")),
    jDim_(context->subVar("j-dim")),
    kDim_(context->subVar("k-dim")),

    iStart_(context->subVar("i-start")),
    jStart_(context->subVar("j-start")),
    kStart_(context->subVar("k-start")),

    iStop_(context->subVar("i-stop")),
    jStop_(context->subVar("j-stop")),
    kStop_(context->subVar("k-stop")),

    iStride_(context->subVar("i-stride")),
    jStride_(context->subVar("j-stride")),
    kStride_(context->subVar("k-stride")),
 
    iWrap_(context->subVar("i-wrap")),
    jWrap_(context->subVar("j-wrap")),
    kWrap_(context->subVar("k-wrap")),

    idim_(0),
    jdim_(0),
    kdim_(0),

    istart_(-1),
    jstart_(-1),
    kstart_(-1),

    istop_(-1),
    jstop_(-1),
    kstop_(-1),

    istride_(10),
    jstride_(5),
    kstride_(1),

    iwrap_(0),
    jwrap_(0),
    kwrap_(0),

    fGeneration_(-1),
    mGeneration_(-1)
{
}

FieldSubSample::~FieldSubSample(){
}

void FieldSubSample::execute(){

  bool updateAll    = false;
  bool updateField  = false;

  FieldHandle fHandle;

  // Get a handle to the input field port.
  FieldIPort* ifield_port =
    (FieldIPort *) get_iport("Input Field");

  if (!ifield_port) {
    error( "Unable to initialize "+name+"'s iport" );
    return;
  }

  // The field input is required.
  if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
      !(fHandle->mesh().get_rep())) {
    error( "No handle or representation" );
    return;
  }

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {
    fGeneration_  = fHandle->generation;

    updateField = true;
  }

  int dims = 0;

  // Get the dimensions of the mesh.
  if( fHandle->get_type_description(0)->get_name() == "LatVolField" ||
      fHandle->get_type_description(0)->get_name() == "StructHexVolField" ) {
    LatVolMesh *lvmInput = (LatVolMesh*) fHandle->mesh().get_rep();

    idim_ = lvmInput->get_ni();
    jdim_ = lvmInput->get_nj();
    kdim_ = lvmInput->get_nk();

    dims = 3;

  } else if( fHandle->get_type_description(0)->get_name() == "ImageField" ||
	     fHandle->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
    ImageMesh *imInput = (ImageMesh*) fHandle->mesh().get_rep();

    idim_ = imInput->get_ni();
    jdim_ = imInput->get_nj();
    kdim_ = 1;

    dims = 2;

  } else if( fHandle->get_type_description(0)->get_name() == "ScanlineField" ||
	     fHandle->get_type_description(0)->get_name() == "StructCurveField" ) {
    ScanlineMesh *slmInput = (ScanlineMesh*) fHandle->mesh().get_rep();

    idim_ = slmInput->get_ni();
    jdim_ = 1;
    kdim_ = 1;

    dims = 1;

  } else {
    error( fHandle->get_type_description(0)->get_name() );
    error( "Only availible for regular topology e.g. uniformly gridded or structure gridded data." );
    return;
  }

  if( fHandle->data_at() != Field::CELL &&
      fHandle->data_at() != Field::NODE ) {
    error( fHandle->get_type_description(0)->get_name() );
    error( "Currently only availible for cell or node data." );
    return;
  }

  int wrap;

  if( fHandle->get_type_description(0)->get_name() == "StructHexVolField" ||
      fHandle->get_type_description(0)->get_name() == "StructQuadSurfField" ||
      fHandle->get_type_description(0)->get_name() == "StructCurveField" )
    wrap = 1;
  else
    wrap = 0;

  // Check to see if the dimensions have changed.
  if( dims  != Dims_.get() ||
      wrap  != Wrap_.get() ||
      idim_ != iDim_.get() ||
      jdim_ != jDim_.get() ||
      kdim_ != kDim_.get() )
  {
    iDim_.set(idim_);
    jDim_.set(jdim_);
    kDim_.set(kdim_);
    Wrap_.set(wrap);
    Dims_.set(dims);

    updateAll = true;
  }

  // Check to see if the user setable values have changed.
  if( istart_ != iStart_.get() ||
      jstart_ != jStart_.get() ||
      kstart_ != kStart_.get() ||

      istop_ != iStop_.get() ||
      jstop_ != jStop_.get() ||
      kstop_ != kStop_.get() ||

      istride_ != iStride_.get() ||
      jstride_ != jStride_.get() ||
      kstride_ != kStride_.get() ||

      iwrap_ != iWrap_.get() ||
      jwrap_ != jWrap_.get() ||
      kwrap_ != kWrap_.get() ) {

    istart_ = iStart_.get();
    jstart_ = jStart_.get();
    kstart_ = kStart_.get();

    istop_ = iStop_.get();
    jstop_ = jStop_.get();
    kstop_ = kStop_.get();

    istride_ = iStride_.get();
    jstride_ = jStride_.get();
    kstride_ = kStride_.get();

    iwrap_ = iWrap_.get();
    jwrap_ = jWrap_.get();
    kwrap_ = kWrap_.get();
    
    updateAll = true;
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || updateAll || updateField ) {

    const TypeDescription *ftd = fHandle->get_type_description();
    CompileInfoHandle ci = FieldSubSampleAlgo::get_compile_info(ftd);
    Handle<FieldSubSampleAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fHandle_ = algo->execute(fHandle,
			     istart_, jstart_, kstart_,
			     istop_, jstop_, kstop_,
			     istride_, jstride_, kstride_,
			     iwrap_, jwrap_, kwrap_);
  }

  // Get a handle to the output field port.
  if( fHandle_.get_rep() ) {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( fHandle_ );
  }
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

  // Structured meshs have a set_point method which is needed. However, it is not
  // defined for gridded meshes. As such, the include file defined below contains a
  // compiler flag so that when needed in FieldSlicer.h it is compiled.
  if( ftd->get_name().find("StructHexVolField"  ) == 0 ||
      ftd->get_name().find("StructQuadSurfField") == 0 ||
      ftd->get_name().find("StructCurveField"   ) == 0 ) {

    string header_path(include_path);  // Get the right path 

    // Insert the Dynamic header file name.
    header_path.insert( header_path.find_last_of("."), "Dynamic" );

    rval->add_include(header_path);
  }

  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
