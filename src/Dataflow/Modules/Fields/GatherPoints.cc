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
 *  GatherPoints.cc: Take in fields and add all of their points into one field
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Modules/Fields/GatherPoints.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <iostream>

namespace SCIRun {

class PSECORESHARE GatherPoints : public Module {
public:
  GatherPoints(GuiContext* ctx);
  virtual ~GatherPoints();
  virtual void execute();
};

DECLARE_MAKER(GatherPoints)
GatherPoints::GatherPoints(GuiContext* ctx)
  : Module("GatherPoints", ctx, Filter, "FieldsCreate", "SCIRun")
{
}

GatherPoints::~GatherPoints()
{
}

void
GatherPoints::execute()
{
  FieldOPort *ofld = (FieldOPort *)get_oport("Field");
  if (!ofld) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  port_range_type range = get_iports("Field");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  PointCloudMeshHandle pcmH = scinew PointCloudMesh;

  while (pi != range.second)
  {
    FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
    if (!ifield) {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }
    FieldHandle field;
    if (ifield->get(field)) {
      const TypeDescription *meshtd = field->mesh()->get_type_description();
      CompileInfoHandle ci = GatherPointsAlgo::get_compile_info(meshtd);
      Handle<GatherPointsAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;
      algo->execute(field->mesh(), pcmH);
    }
    ++pi;
  }
  PointCloudField<double> *pcH = scinew PointCloudField<double>(pcmH, Field::NODE);
  ofld->send(FieldHandle(pcH));
}

CompileInfoHandle
GatherPointsAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GatherPointsAlgoT");
  static const string base_class_name("GatherPointsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
