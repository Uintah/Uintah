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
  if (fields.size())
  {
    const TypeDescription *td = fields[0]->get_type_description();
    CompileInfoHandle ci = GatherFieldsAlgo::get_compile_info(td);
    Handle<GatherFieldsAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;
    ofield = algo->execute(fields);
  }

  FieldOPort *ofld = (FieldOPort *)get_oport("Field");
  if (!ofld) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  ofld->send(ofield);
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
