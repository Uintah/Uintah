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
 *  MeshBuilde.cc:
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
#include <Packages/Fusion/Dataflow/Modules/Fields/FusionSlicer.h>
#include <Packages/Fusion/Core/Datatypes/StructHexVolField.h>
#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE FusionSlicer : public Module {
public:
  FusionSlicer(const string& id);

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
  int mGeneration_;
};

extern "C" FusionSHARE Module* make_FusionSlicer(const string& id) {
  return scinew FusionSlicer(id);
}

FusionSlicer::FusionSlicer(const string& id)
  : Module("FusionSlicer", id, Source, "Fields", "Fusion"),
    
    Axis_("axis", id, this),

    iDim_("idim", id, this),
    jDim_("jdim", id, this),
    kDim_("kdim", id, this),

    iIndex_("iindex", id, this),
    jIndex_("jindex", id, this),
    kIndex_("kindex", id, this),

    axis_(2),

    idim_(0),
    jdim_(0),
    kdim_(0),

    iindex_(-1),
    jindex_(-1),
    kindex_(-1),

    fGeneration_(-1),
    mGeneration_(-1)

{
}

FusionSlicer::~FusionSlicer(){
}

void FusionSlicer::execute(){

  bool updateAll    = false;
  bool updateField  = false;

  FieldHandle fHandle;

  StructHexVolMesh *hvmInput;
  StructHexVolField<double> *hvfInput;

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
  hvfInput = (StructHexVolField<double> *)(fHandle.get_rep());

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
  if( idim_+1 != iDim_.get() ||
      jdim_ != jDim_.get() ||
      kdim_ != kDim_.get() )
  {
    // Update the dims in the GUI.
    ostringstream str;
    str << id << " set_size " << idim_+1 << " " << jdim_ << " " << kdim_;

    TCL::execute(str.str().c_str());

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

    const TypeDescription *ftd = fHandle->get_type_description();
    CompileInfo *ci = FusionSlicerAlgo::get_compile_info(ftd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      cout << "Could not compile algorithm." << std::endl;
      return;
    }
    FusionSlicerAlgo *algo =
      dynamic_cast<FusionSlicerAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      cout << "Could not get algorithm." << std::endl;
      return;
    }
    unsigned int index;
    if (axis_ == 0)
    {
      index = Max(iindex_, 1);
    }
    else if (axis_ == 1)
    {
      index = jindex_;
    }
    else
    {
      index = kindex_;
    }
    fHandle_ = algo->execute(fHandle, index, axis_);
  }

  // Get a handle to the output field port.
  if( fHandle_.get_rep() )
  {
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
FusionSlicerAlgo::get_compile_info(const TypeDescription *fld_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FusionSlicerAlgoT");
  static const string base_class_name("FusionSlicerAlgo");
  const string::size_type fld_loc = fld_td->get_name().find_first_of('<');
  const string fdst = "QuadSurfField" + fld_td->get_name().substr(fld_loc);

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fld_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fld_td->get_name() + ", " + fdst);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fld_td->fill_compile_info(rval);
  return rval;
}


} // End namespace Fusion
