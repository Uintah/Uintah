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
 *  GatherFields.cc: Take in fields and append them into one field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   July 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Modules/Fields/GatherFields.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <iostream>

namespace SCIRun {

class PSECORESHARE GatherFields : public Module {
public:
  GatherFields(GuiContext* ctx);
  virtual ~GatherFields();
  virtual void execute();
};

DECLARE_MAKER(GatherFields)
GatherFields::GatherFields(GuiContext* ctx)
  : Module("GatherFields", ctx, Filter, "FieldsCreate", "SCIRun")
{
}


GatherFields::~GatherFields()
{
}


void
GatherFields::execute()
{
  unsigned int i;

  port_range_type range = get_iports("Field");
  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;

  // Gather up all of the field handles.
  vector<FieldHandle> fields;
  int counter = 0;
  while (pi != range.second)
  {
    FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
    if (!ifield) {
      error("Unable to initialize iport '" + to_string(pi->second) + "'.");
      return;
    }
    // Increment here!  We do this because last one is always
    // empty so we can test for it before issuing empty warning.
    ++pi;
    FieldHandle field;
    if (ifield->get(field) && field.get_rep())
    {
      fields.push_back(field);
    }
    else if (pi != range.second)
    {
      warning("Input port " + to_string(counter+1) + " contained no data.");
    }
    ++counter;
  }

  FieldHandle ofield(0);
  if (fields.size() == 1)
  {
    ofield = fields[0];
  }
  else if (fields.size() > 1)
  {
    const TypeDescription *mtd0 = fields[0]->mesh()->get_type_description();
    const TypeDescription *ftd0 = fields[0]->get_type_description();
    const int loc0 = fields[0]->basis_order();
    bool same_field_kind = true;
    bool same_mesh_kind = true;
    bool same_data_location = true;
    for (i = 1; i < fields.size(); i++)
    {
      if (fields[i]->mesh()->get_type_description() != mtd0)
      {
	same_mesh_kind = false;
      }
      if (fields[i]->get_type_description() != ftd0)
      {
	same_field_kind = false;
      }
      if (fields[i]->basis_order() != loc0)
      {
	same_data_location = false;
      }
    }

    if (fields[0]->mesh()->is_editable() &&
	(same_field_kind || same_mesh_kind))
    {
      bool copy_data = same_data_location;
      if (!same_data_location)
      {
	warning("Cannot copy data from meshes with different data locations.");
      }
      else if (!same_field_kind)
      {
	warning("Copying data does not work for data of different kinds.");
	copy_data = false;
      }
      else if (same_data_location && fields[0]->basis_order() != 1)
      {
	warning("Copying data does not work for non-node data locations.");
	copy_data = false;
      }
      CompileInfoHandle ci = GatherFieldsAlgo::get_compile_info(ftd0);
      Handle<GatherFieldsAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;
      ofield = algo->execute(fields, copy_data);
    }
    else
    {
      if (same_field_kind || same_mesh_kind)
      {
	warning("Non-editable meshes detected, outputting PointCloudField.");
      }
      else
      {
	warning("Different mesh types detected, outputting PointCloudField.");
      }
      PointCloudMeshHandle pc = scinew PointCloudMesh;
      for (i = 0; i < fields.size(); i++)
      {
	const TypeDescription *mtd = fields[i]->mesh()->get_type_description();
	CompileInfoHandle ci = GatherPointsAlgo::get_compile_info(mtd);
	Handle<GatherPointsAlgo> algo;
	if (!module_dynamic_compile(ci, algo)) return;
	algo->execute(fields[i]->mesh(), pc);
      }
      ofield = scinew PointCloudField<double>(pc, 1);
    }
  }

  FieldOPort *ofld = (FieldOPort *)get_oport("Field");
  if (!ofld) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  ofld->send(ofield);
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


CompileInfoHandle
GatherFieldsAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GatherFieldsAlgoT");
  static const string base_class_name("GatherFieldsAlgo");

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
