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
 *  ChangeCoordinates.cc: Take in fields and add all of their points into one field
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
#include <Dataflow/Modules/Fields/ChangeCoordinates.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/DynamicCompilation.h>
#include <iostream>

namespace SCIRun {

class PSECORESHARE ChangeCoordinates : public Module {
public:
  ChangeCoordinates(GuiContext* ctx);
  virtual ~ChangeCoordinates();
  virtual void execute();
  GuiString gui_oldsystem_;
  GuiString gui_newsystem_;
};

DECLARE_MAKER(ChangeCoordinates)
ChangeCoordinates::ChangeCoordinates(GuiContext* ctx)
  : Module("ChangeCoordinates", ctx, Filter, "FieldsGeometry", "SCIRun"),
    gui_oldsystem_(ctx->subVar("oldsystem")),
    gui_newsystem_(ctx->subVar("newsystem"))
{
}

ChangeCoordinates::~ChangeCoordinates()
{
}

void
ChangeCoordinates::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 
  if (!iport) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  
  // The input port (with data) is required.
  FieldHandle field;
  if (!iport->get(field) || !field.get_rep())
  {
    error("No input data");
    return;
  }

  FieldOPort *ofld = (FieldOPort *)get_oport("Output Field");
  if (!ofld) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  string oldsystem=gui_oldsystem_.get();
  string newsystem=gui_newsystem_.get();
  field.detach();
  field->mesh_detach();
  const TypeDescription *meshtd = field->mesh()->get_type_description();
  CompileInfoHandle ci = ChangeCoordinatesAlgo::get_compile_info(meshtd);
  Handle<ChangeCoordinatesAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;
  algo->execute(field->mesh(), oldsystem, newsystem);
  ofld->send(field);
}

CompileInfo *
ChangeCoordinatesAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ChangeCoordinatesAlgoT");
  static const string base_class_name("ChangeCoordinatesAlgo");

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
