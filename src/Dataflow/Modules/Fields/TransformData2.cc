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
 *  TransformData2: Unary field data operations
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
#include <Dataflow/Modules/Fields/TransformData2.h>
#include <Core/Util/DynamicCompilation.h>

#include <iostream>
#include <sci_hash_map.h>

namespace SCIRun {

class TransformData2 : public Module
{
private:
  GuiString function_;
  GuiString outputdatatype_;

public:
  TransformData2(GuiContext* ctx);
  virtual ~TransformData2();
  virtual void execute();
};


DECLARE_MAKER(TransformData2)


TransformData2::TransformData2(GuiContext* ctx)
  : Module("TransformData2", ctx, Filter,"Fields", "SCIRun"),
    function_(ctx->subVar("function")),
    outputdatatype_(ctx->subVar("outputdatatype"))
{
}


TransformData2::~TransformData2()
{
}


void
TransformData2::execute()
{
  // Get input field.
  FieldIPort *ifp0 = (FieldIPort *)get_iport("Input Field 0");
  FieldHandle ifieldhandle0;
  if (!ifp0) {
    error("Unable to initialize iport 'Input Field 0'.");
    return;
  }
  if (!(ifp0->get(ifieldhandle0) && ifieldhandle0.get_rep()))
  {
    error("Input field 0 is empty.");
    return;
  }

  if (ifieldhandle0->data_at() == Field::NONE)
  {
    error("Field 0 contains no data to transform.");
    return;
  }

  FieldIPort *ifp1 = (FieldIPort *)get_iport("Input Field 1");
  FieldHandle ifieldhandle1;
  if (!ifp1) {
    error("Unable to initialize iport 'Input Field 1'.");
    return;
  }
  if (!(ifp1->get(ifieldhandle1) && ifieldhandle1.get_rep()))
  {
    error("Input field 1 is empty.");
    return;
  }

  if (ifieldhandle1->data_at() == Field::NONE)
  {
    error("Field 1 contains no data to transform.");
    return;
  }

  if (ifieldhandle0->mesh().get_rep() != ifieldhandle1->mesh().get_rep())
  {
    error("The Input Fields must share the same mesh.");
    return;
  }

  if (ifieldhandle0->data_at() != ifieldhandle1->data_at())
  {
    error("The Input Fields must share the same data location.");
    return;
  }

  string outputdatatype = outputdatatype_.get();
  if (outputdatatype == "input 0")
  {
    outputdatatype = ifieldhandle0->get_type_description(1)->get_name();
  }
  else if (outputdatatype == "input 1")
  {
    outputdatatype = ifieldhandle1->get_type_description(1)->get_name();
  }

  const TypeDescription *ftd0 = ifieldhandle0->get_type_description();
  const TypeDescription *ftd1 = ifieldhandle1->get_type_description();
  const TypeDescription *ltd = ifieldhandle0->data_at_type_description();
  const string oftn = ifieldhandle0->get_type_description(0)->get_name() +
    "<" + outputdatatype + "> ";
  int hoffset = 0;
  Handle<TransformData2Algo> algo;
  while (1)
  {
    CompileInfoHandle ci =
      TransformData2Algo::get_compile_info(ftd0, ftd1, oftn, ltd,
					   function_.get(), hoffset);
    if (!DynamicCompilation::compile(ci, algo, true, this))
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

  FieldHandle ofieldhandle = algo->execute(ifieldhandle0, ifieldhandle1);

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformData2Algo::get_compile_info(const TypeDescription *field0_td,
				     const TypeDescription *field1_td,
				     string ofieldtypename,
				     const TypeDescription *loc_td,
				     string function,
				     int hashoffset)

{
  hash<const char *> H;
  unsigned int hashval = H(function.c_str()) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("TransformData2Instance" + to_string(hashval));
  static const string base_class_name("TransformData2Algo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field0_td->get_filename() + "." +
		       field1_td->get_filename() + "." +
		       to_filename(ofieldtypename) + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       field0_td->get_name() + ", " +
                       field1_td->get_name() + ", " +
		       ofieldtypename + ", " +
		       loc_td->get_name());

  // Code for the function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "template <class IFIELD0, class IFIELD1, class OFIELD, class LOC>\n" +
    "class " + template_name + " : public TransformData2AlgoT<IFIELD0, IFIELD1, OFIELD, LOC>\n" +
    "{\n" +
    "  virtual void function(typename OFIELD::value_type &result,\n" +
    "                        double x, double y, double z,\n" +
    "                        const typename IFIELD0::value_type &v0,\n" +
    "                        const typename IFIELD1::value_type &v1)\n" +
    "  {\n" +
    "    " + function + "\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + function + "\"); }\n" +
    "};\n//";

  // Add in the include path to compile this obj
  rval->add_include(include_path + class_declaration);
  field0_td->fill_compile_info(rval);
  field1_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
