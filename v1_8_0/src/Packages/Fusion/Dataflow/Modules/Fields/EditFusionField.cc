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
 *  EditFusionField.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Fusion/Dataflow/Modules/Fields/EditFusionField.h>
#include <Packages/Fusion/Core/Datatypes/StructHexVolField.h>
#include <Packages/Fusion/share/share.h>
#include <Core/Containers/Handle.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE EditFusionField : public Module {
public:
  EditFusionField(GuiContext *context);

  virtual ~EditFusionField();

  virtual void execute();

private:
  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iStart_;
  GuiInt jStart_;
  GuiInt kStart_;

  GuiInt iDelta_;
  GuiInt jDelta_;
  GuiInt kDelta_;

  GuiInt iSkip_;
  GuiInt jSkip_;
  GuiInt kSkip_;

  int idim_;
  int jdim_;
  int kdim_;

  int istart_;
  int jstart_;
  int kstart_;

  int iend_;
  int jend_;
  int kend_;

  int iskip_;
  int jskip_;
  int kskip_;

  FieldHandle  fHandle_;

  int fGeneration_;
  int mGeneration_;
};


DECLARE_MAKER(EditFusionField)


EditFusionField::EditFusionField(GuiContext *context)
  : Module("EditFusionField", context, Source, "Fields", "Fusion"),

    iDim_(context->subVar("idim")),
    jDim_(context->subVar("jdim")),
    kDim_(context->subVar("kdim")),

    iStart_(context->subVar("istart")),
    jStart_(context->subVar("jstart")),
    kStart_(context->subVar("kstart")),

    iDelta_(context->subVar("idelta")),
    jDelta_(context->subVar("jdelta")),
    kDelta_(context->subVar("kdelta")),

    iSkip_(context->subVar("iskip")),
    jSkip_(context->subVar("jskip")),
    kSkip_(context->subVar("kskip")),

    idim_(0),
    jdim_(0),
    kdim_(0),

    istart_(-1),
    jstart_(-1),
    kstart_(-1),

    iend_(-1),
    jend_(-1),
    kend_(-1),

    iskip_(10),
    jskip_(5),
    kskip_(1),

    fGeneration_(-1),
    mGeneration_(-1)
{
}

EditFusionField::~EditFusionField(){
}

void EditFusionField::execute(){

  bool updateAll    = false;
  bool updateField  = false;

  FieldHandle fHandle;

  StructHexVolMesh *hvmInput;
  StructHexVolField<double> *hvfInput;

  // Get a handle to the input field port.
  FieldIPort* ifield_port =
    (FieldIPort *)	get_iport("Input Field");

  if (!ifield_port) {
    error( "Unable to initialize "+name+"'s iport" );
    return;
  }

  // The field input is required.
  if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
      !(hvmInput = (StructHexVolMesh*) fHandle->mesh().get_rep())) {
    error( "No handle or representation" );
    return;
  }
  hvfInput = (StructHexVolField<double> *)(fHandle.get_rep());

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {
    fGeneration_  = fHandle->generation;

    updateField = true;
  }

  // Get the dimensions of the mesh.
  idim_ = hvmInput->get_nx();
  jdim_ = hvmInput->get_ny();
  kdim_ = hvmInput->get_nz();

  // Check to see if the dimensions have changed.
  if( idim_   != iDim_.get() ||
      jdim_-1 != jDim_.get() ||
      kdim_-1 != kDim_.get() ) {

    // Update the dims in the GUI.
    ostringstream str;
    str << id << " set_size " << idim_ << " " << jdim_-1 << " " << kdim_-1;

    gui->execute(str.str().c_str());

    updateAll = true;
  }


  // Check to see if the user setable values have changed.
  if( istart_ != iStart_.get() ||
      jstart_ != jStart_.get() ||
      kstart_ != kStart_.get() ||

      iend_  != (iStart_.get() + iDelta_.get()) ||
      jend_  != (jStart_.get() + jDelta_.get()) ||
      kend_  != (kStart_.get() + kDelta_.get()) ||

      iskip_ != iSkip_.get()  ||
      jskip_ != jSkip_.get() ||
      kskip_ != kSkip_.get() ) {

    istart_ = iStart_.get();
    jstart_ = jStart_.get();
    kstart_ = kStart_.get();

    iend_ = iStart_.get() + iDelta_.get();
    jend_ = jStart_.get() + jDelta_.get();
    kend_ = kStart_.get() + kDelta_.get();

    iskip_ = iSkip_.get();
    jskip_ = jSkip_.get();
    kskip_ = kSkip_.get();

    updateAll = true;
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || updateAll || updateField ) {

    const TypeDescription *ftd = fHandle->get_type_description();
    CompileInfo *ci = EditFusionFieldAlgo::get_compile_info(ftd);
    Handle<EditFusionFieldAlgo> algo;
    if (!module_dynamic_compile(*ci, algo)) return;

    fHandle_ = algo->execute(fHandle,
			     istart_, jstart_, kstart_,
			     iend_, jend_, kend_,
			     iskip_, jskip_, kskip_);

    delete ci;
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

CompileInfo *
EditFusionFieldAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("EditFusionFieldAlgoT");
  static const string base_class_name("EditFusionFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace Fusion
