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
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Containers/Handle.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/FusionSlicePlot.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE FusionSlicePlot : public Module {
public:
  FusionSlicePlot(GuiContext *context);

  virtual ~FusionSlicePlot();

  virtual void execute();

private:
  GuiDouble Scale_;

  double scale_;

  FieldHandle  fHandle_;

  int fGeneration_;
};


DECLARE_MAKER(FusionSlicePlot)


FusionSlicePlot::FusionSlicePlot(GuiContext *context)
  : Module("FusionSlicePlot", context, Source, "Fields", "Fusion"),
    
    Scale_(context->subVar("scale")),

    scale_(0.0),

    fGeneration_(-1)
{
}

FusionSlicePlot::~FusionSlicePlot(){
}

void FusionSlicePlot::execute(){

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

  if (fHandle->query_scalar_interface(this).get_rep() == 0)
  {
    error("This module only works on scalar fields.");
    return;
  }

  if (fHandle->data_at() != Field::NODE) {
    error("This module only works for fields with data at nodes.");
    return;
  }

  const TypeDescription *ftd = fHandle->get_type_description();

  string sname = ftd->get_name("", "");
  if (sname.find("TetVolField") == string::npos &&
      sname.find("HexVolField") == string::npos &&
      sname.find("TriSurfField") == string::npos &&
      sname.find("QuadSurfField") == string::npos &&
      sname.find("CurveField") == string::npos &&
      sname.find("PointCloudField") == string::npos) {
    error("Can not change coordinates of this field (mesh) type.");
    return;
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() ||
      fGeneration_ != fHandle->generation ||
      scale_ != Scale_.get() ) {

    fGeneration_  = fHandle->generation;
    scale_ = Scale_.get();

    CompileInfoHandle ci = FusionSlicePlotAlgo::get_compile_info(ftd);
    Handle<FusionSlicePlotAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fHandle_ = algo->execute(fHandle, scale_);

    if( fHandle_.get_rep() == NULL ) {
      error( "Only availible for Scalar data." );
      return;
    }
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
FusionSlicePlotAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FusionSlicePlotAlgoT");
  static const string base_class_name("FusionSlicePlotAlgo");

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
