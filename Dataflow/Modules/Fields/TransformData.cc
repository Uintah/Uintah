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
 *  TransformData: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformData.h>

#include <iostream>
#include <sci_hash_map.h>

namespace SCIRun {

class TransformData : public Module
{
private:
  GuiString function_;

public:
  TransformData(GuiContext* ctx);
  virtual ~TransformData();
  virtual void execute();
};


DECLARE_MAKER(TransformData)


TransformData::TransformData(GuiContext* ctx)
  : Module("TransformData", ctx, Filter,"Fields", "SCIRun"),
    function_(ctx->subVar("function"))
{
}


TransformData::~TransformData()
{
}


void
TransformData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Input field is empty.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  int hoffset = 0;
  Handle<TransformDataAlgo> algo;
  while (1)
  {
    CompileInfoHandle ci =
      TransformDataAlgo::get_compile_info(ftd, ltd, function_.get(), hoffset);
    if (!module_maybe_dynamic_compile(ci, algo))
    {
      DynamicLoader::scirun_loader().remove_cc(*(ci.get_rep()), cout);
      error("Your function would not compile.");
      return;
    }
    if (algo->identify() == function_.get())
    {
      break;
    }
    hoffset++;
  }

  FieldHandle ofieldhandle = algo->execute(ifieldhandle);

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformDataAlgo::get_compile_info(const TypeDescription *field_td,
				    const TypeDescription *loc_td,
				    string function,
				    int hashoffset)
{
  hash<const char *> H;
  unsigned int hashval = H(function.c_str()) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("TransformDataInstance" + to_string(hashval));
  static const string base_class_name("TransformDataAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       field_td->get_name() + ", " + loc_td->get_name());

  // Code for the function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "template <class FIELD, class LOC>\n" +
    "class " + template_name + " : public TransformDataAlgoT<FIELD, LOC>\n" +
    "{\n" +
    "  virtual void function(typename FIELD::value_type &result,\n" +
    "                        double x, double y, double z,\n" +
    "                        const typename FIELD::value_type &v)\n" +
    "  {\n" +
    function +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + function + "\"); }\n" +
    "};\n//";

  // Add in the include path to compile this obj
  rval->add_include(include_path + class_declaration);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
