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
 *  MapDataToMeshCoord.cc:  Replace a mesh coordinate with the fdata values
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Modules/Fields/MapDataToMeshCoord.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


class MapDataToMeshCoord : public Module
{
  GuiInt gui_coord_;
public:
  MapDataToMeshCoord(GuiContext* ctx);
  virtual ~MapDataToMeshCoord();

  virtual void execute();
protected:
  int last_generation_;
  int last_gui_coord_;
};


DECLARE_MAKER(MapDataToMeshCoord)

MapDataToMeshCoord::MapDataToMeshCoord(GuiContext* ctx)
  : Module("MapDataToMeshCoord", ctx, Filter, "FieldsGeometry", "SCIRun"),
    gui_coord_(ctx->subVar("coord")), last_generation_(0)
{
}

MapDataToMeshCoord::~MapDataToMeshCoord()
{
}

void
MapDataToMeshCoord::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifield;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifield) && ifield.get_rep()))
  {
    error("Input field is empty.");
    return;
  }

  if (ifield->query_scalar_interface(this).get_rep() == 0)
  {
    error("This module only works on scalar fields.");
    return;
  }
  if (ifield->data_at() != Field::NODE) {
    error("This module only works for fields with data at nodes.");
    return;
  }

  const TypeDescription *ftd = ifield->get_type_description();
  string sname = ftd->get_name("", "");
  if (sname.find("TetVolField") == string::npos &&
      sname.find("HexVolField") == string::npos &&
      sname.find("TriSurfField") == string::npos &&
      sname.find("QuadSurfField") == string::npos &&
      sname.find("CurveField") == string::npos &&
      sname.find("PointCloudField") == string::npos) {
    error("Can't change coordinates of this field (mesh) type.");
    return;
  }

  int coord = gui_coord_.get();
  if (coord == 3) {
    if (sname.find("TriSurfField") == string::npos &&
	sname.find("QuadSurfField") == string::npos) {
      error("Can't get a normal from this type of mesh.");
      return;
    }
  }
      
  if (last_generation_ != ifield->generation || last_gui_coord_ != coord) {
    last_generation_ = ifield->generation;
    last_gui_coord_ = coord;

    CompileInfoHandle ci = MapDataToMeshCoordAlgo::get_compile_info(ftd);

    Handle<MapDataToMeshCoordAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;
    
    FieldHandle ofield(algo->execute(ifield, coord));
    
    FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
    if (!ofp) {
      error("Unable to initialize oport 'Output Field'.");
      return;
    }

    ofp->send(ofield);
  }
}

CompileInfoHandle
MapDataToMeshCoordAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("MapDataToMeshCoordAlgoT");
  static const string base_class_name("MapDataToMeshCoordAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
