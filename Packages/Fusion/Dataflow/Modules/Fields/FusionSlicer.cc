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
 *  FusionSlicer.cc:
 *
 *  Written by:
 *   Michael Callahan
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/StructHexVolField.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/FusionSlicer.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE FusionSlicer : public Module {
public:
  FusionSlicer(GuiContext *context);

  virtual ~FusionSlicer();

  virtual void execute();

private:
  GuiInt Axis_;

  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iIndex_;
  GuiInt jIndex_;
  GuiInt kIndex_;

  int axis_;

  int idim_;
  int jdim_;
  int kdim_;

  int iindex_;
  int jindex_;
  int kindex_;

  FieldHandle  fHandle_;

  int fGeneration_;
};


DECLARE_MAKER(FusionSlicer)


FusionSlicer::FusionSlicer(GuiContext *context)
  : Module("FusionSlicer", context, Source, "Fields", "Fusion"),
    
    Axis_(context->subVar("axis")),

    iDim_(context->subVar("idim")),
    jDim_(context->subVar("jdim")),
    kDim_(context->subVar("kdim")),

    iIndex_(context->subVar("iindex")),
    jIndex_(context->subVar("jindex")),
    kIndex_(context->subVar("kindex")),

    axis_(2),

    idim_(0),
    jdim_(0),
    kdim_(0),

    iindex_(-1),
    jindex_(-1),
    kindex_(-1),

    fGeneration_(-1)
{
}

FusionSlicer::~FusionSlicer(){
}

void FusionSlicer::execute(){

  bool updateAll    = false;
  bool updateField  = false;

  FieldHandle fHandle;

  StructHexVolMesh *hvmInput;

  // Get a handle to the input field port.
  FieldIPort* ifield_port =
    (FieldIPort *)	get_iport("Input Field");

  if (!ifield_port)
  {
    error( "Unable to initialize "+name+"'s iport" );
    return;
  }

  // The field input is required.
  if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
      !(hvmInput = (StructHexVolMesh*) fHandle->mesh().get_rep()))
  {
    error( "No handle or representation" );
    return;
  }

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation )
  {
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
      kdim_-1 != kDim_.get() )
  {
    // Update the dims in the GUI.
    ostringstream str;
    str << id << " set_size " << idim_ << " " << jdim_-1 << " " << kdim_-1;

    gui->execute(str.str().c_str());

    updateAll = true;
  }

  // Check to see if the user setable values have changed.
  if( iindex_ != iIndex_.get() ||
      jindex_ != jIndex_.get() ||
      kindex_ != kIndex_.get() ||
      axis_ != Axis_.get())
  {
    iindex_ = iIndex_.get();
    jindex_ = jIndex_.get();
    kindex_ = kIndex_.get();

    axis_ = Axis_.get();

    updateAll = true;
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || updateAll || updateField )
  {
    const TypeDescription *ftd = fHandle->get_type_description(0);
    const TypeDescription *ttd = fHandle->get_type_description(1);

    CompileInfo *ci = FusionSlicerAlgo::get_compile_info(ftd,ttd);
    Handle<FusionSlicerAlgo> algo;
    if (!module_dynamic_compile(*ci, algo)) return;

    unsigned int index;
    if (axis_ == 0) {
      index = Max(iindex_, 1);
    }
    else if (axis_ == 1) {
      index = jindex_;
    }
    else {
      index = kindex_;
    }
    fHandle_ = algo->execute(fHandle, index, axis_);
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
FusionSlicerAlgo::get_compile_info(const TypeDescription *ftd,
				   const TypeDescription *ttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FusionSlicerAlgoT");
  static const string base_class_name("FusionSlicerAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " +
		       ttd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace Fusion
